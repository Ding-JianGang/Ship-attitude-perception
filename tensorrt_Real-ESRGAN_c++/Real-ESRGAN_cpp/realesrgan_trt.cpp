#include "realesrgan_trt.h"


int RealESRGan_Trt::init_model(const std::string& engine_file_path) {

	cudaSetDevice(m_device);
    static Logger gLogger;
    char* trtModelStream{ nullptr };
    size_t size{ 0 };

    std::ifstream file(engine_file_path, std::ios::binary);

    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        if (!trtModelStream)
            return -2;
        file.read(trtModelStream, size);
        file.close();
    }
    
    m_runtime = nvinfer1::createInferRuntime(gLogger);
    if (m_runtime == nullptr)  return -1;
    m_engine = m_runtime->deserializeCudaEngine(trtModelStream, size);
    if (m_engine == nullptr)  return -1;
    m_context = m_engine->createExecutionContext();
    if (m_context == nullptr)  return -1;

    delete[] trtModelStream;;

    return 1;
}

/*
模型推理
*/
void RealESRGan_Trt::doInference(nvinfer1::IExecutionContext& context, float* input1, float* output, const int output_size, cv::Size input_shape) {
    const nvinfer1::ICudaEngine& engine = context.getEngine();

    
    //assert(engine.getNbBindings() == 2);
    void* buffers[2];   //指向输入缓冲区和输出缓冲区

    const int inputIndex = engine.getBindingIndex("input");  //通过输入层的名称确定输入index  名称在pth转换成onnx时已经确定
    //assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);

    const int outputIndex = engine.getBindingIndex("output");    //通过输出层的名称确定输入index   名称在pth转换成onnx时已经确定
    //assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine.getMaxBatchSize();

    //-------------------------------for dynamic shapes-------------------------
    context.setOptimizationProfile(0);    //for dynamic shapes
    nvinfer1::Dims inputDim;
    inputDim.nbDims = 4;
    inputDim.d[0] = 1;
    inputDim.d[1] = 3;
    inputDim.d[2] = input_shape.height;
    inputDim.d[3] = input_shape.width;
    context.setBindingDimensions(inputIndex, inputDim);     //for dynamic shapes
    //---------------------------------------------------------------------------
    // 在GPU上分配空间
    CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], output_size * sizeof(float)));

    // 创建流
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // 推理
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input1, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));  
    context.enqueue(1, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // 释放
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}



cv::Mat RealESRGan_Trt::infer_Net(cv::Mat& srcImg1, const int& upSize) {

    cv::Mat result;
    if (srcImg1.channels() == 1) cv::cvtColor(srcImg1, srcImg1, cv::COLOR_GRAY2BGR);
    if (srcImg1.channels() != 3)  return result;    //图像格式不符合3通道要求

    auto w = srcImg1.cols, h = srcImg1.rows;
    int w_upsize = w * upSize, h_upsize = h * upSize;
    result = cv::Mat(h_upsize, w_upsize, CV_8UC3);

    int xtiles = std::ceil(w * 1.0 / m_tile_size);
    int ytiles = std::ceil(h * 1.0 / m_tile_size);

    int bottom = m_tile_size * ytiles - h;
    int right = m_tile_size * xtiles - w;
    int prepadding = 10;   //same as source code

    cv::Mat makeBoder_img;
    cv::copyMakeBorder(srcImg1, makeBoder_img, prepadding, bottom, prepadding, right, CV_HAL_BORDER_REFLECT);

    for (int yi = 0; yi < ytiles; yi++) {
        for (int xi = 0; xi < xtiles; xi++) {

            int ofs_x = xi * m_tile_size;
            int ofs_y = yi * m_tile_size;

            int input_start_x = ofs_x;
            int input_end_x = ofs_x + m_tile_size;
            int input_start_y = ofs_y;
            int input_end_y = ofs_y + m_tile_size;

            int input_start_x_pad = input_start_x - prepadding;
            int input_end_x_pad = input_end_x + prepadding;
            int input_start_y_pad = input_start_y - prepadding;
            int input_end_y_pad = input_end_y + prepadding;
            
            if (xi == xtiles - 1) {
                input_end_x_pad = input_end_x;
                if(xi != 0)
                    input_start_x_pad -= prepadding;
            }
            if (yi == ytiles - 1) {
                input_end_y_pad = input_end_y;
                if(yi != 0)
                    input_start_y_pad -= prepadding;
            }
          
            int input_start_x_border = input_start_x_pad + prepadding;
            int input_end_x_border = input_end_x_pad + prepadding;
            int input_start_y_border = input_start_y_pad + prepadding;
            int input_end_y_border = input_end_y_pad + prepadding;

            if (xi == xtiles - 1) {
                assert(input_end_x == makeBoder_img.cols - 1);
                assert(input_end_x_border - input_start_x_border == m_tile_size + 2 * prepadding);
            }
            if (yi == ytiles - 1) {
                assert(input_end_y == makeBoder_img.rows - 1);
                assert(input_end_y_border - input_start_y_border == m_tile_size + 2 * prepadding);
            }
            
            cv::Mat input_tile = makeBoder_img(cv::Rect(input_start_x_border, input_start_y_border, input_end_x_border - input_start_x_border, input_end_y_border - input_start_y_border));
            cv::Mat intput_Mat = cv::dnn::blobFromImage(input_tile, 1.0 / 255, cv::Size(), cv::Scalar(), true);
            
            int width_up = input_tile.cols * upSize;
            int height_up = input_tile.rows * upSize;

            int img_len = width_up * height_up;

            int output_size = 1 * 3 * img_len;

            float* output = new float[output_size];    //初始化输出
            auto start = std::chrono::system_clock::now();
            try {

                doInference(*m_context, (float*)intput_Mat.data, output, output_size, cv::Size(input_tile.cols, input_tile.rows));   //使用gpu推理 

            }
            catch (const std::exception& ex) {
                std::cout << "推理错误：" << ex.what() << std::endl;
                return result;
            }

            auto end = std::chrono::system_clock::now();
            std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

            int prepadding_upSize = prepadding * upSize;
            int head_j = prepadding_upSize, head_i = prepadding_upSize;
            int tail_j = prepadding_upSize, tail_i = prepadding_upSize;

            if (xi == xtiles - 1) {
                tail_j = right * upSize;
                if(xi == 0)
                    head_j = prepadding_upSize;
                else
                    head_j = prepadding_upSize * 2;
            }
            if (yi == ytiles - 1) {
                tail_i = bottom * upSize;
                if(yi == 0)
                    head_i = prepadding_upSize;
                else
                    head_i = prepadding_upSize * 2;
            }
            int tile_size_up = m_tile_size * upSize;

            for (int i = head_i; i < height_up - tail_i; i++) {
                for (int j = head_j; j < width_up - tail_j; j++) {
                    int img_idx_row = yi * tile_size_up + i - head_i;
                    int img_idx_col = xi * tile_size_up + j - head_j;
                    result.ptr<uchar>(img_idx_row, img_idx_col)[2] = cv::saturate_cast<uchar>(output[i * width_up + j] * 255);
                    result.ptr<uchar>(img_idx_row, img_idx_col)[1] = cv::saturate_cast<uchar>(output[img_len + i * width_up + j] * 255);
                    result.ptr<uchar>(img_idx_row, img_idx_col)[0] = cv::saturate_cast<uchar>(output[img_len * 2 + i * width_up + j] * 255);
                }
            }

            delete output;

        }
    }

    return result;
}


