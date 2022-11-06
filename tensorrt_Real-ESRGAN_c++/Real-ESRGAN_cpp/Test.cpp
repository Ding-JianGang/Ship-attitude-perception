#include "realesrgan_trt.h"


int main() {

	std::string imgFile1 = "D:\\tmp_test\\source_0.jpg";
	std::string engineFile = "./realesrgan-x4_64-276.engine";

	cv::Mat img1 = cv::imread(imgFile1);
	int tile_size = 256;
	int upSize = 4;    //4 or 2

	RealESRGan_Trt* rt = new RealESRGan_Trt(0, tile_size);
	int init_code = rt->init_model(engineFile);

	if (init_code != 1) {
		std::cout << "模型初始化失败, " << init_code << std::endl;
		return 0;
	}

	auto start = std::chrono::system_clock::now();
	cv::Mat result = rt->infer_Net(img1, upSize);
	std::cout << "用时：";
	auto end = std::chrono::system_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	std::cout << result.size << std::endl;
	cv::namedWindow("result", 0);
	cv::imshow("result", result);
	cv::waitKey(0);

	return 1;
}


