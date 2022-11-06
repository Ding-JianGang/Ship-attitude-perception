import os
import sys
import glob
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from labelme import utils
from sklearn.model_selection import train_test_split


class Labelme2coco():
    def __init__(self, args):
        self.classname_to_id = {args.class_name: 1}
        self.images = []
        self.annotations = []
        self.categories = []
        self.ann_id = 0

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def _get_keypoints(self, points, keypoints, num_keypoints):
        if points[0] == 0 and points[1] == 0:
            visable = 0
        else:
            visable = 2
            num_keypoints += 1
        keypoints.extend([points[0], points[1], visable])
        return keypoints, num_keypoints

    def _image(self, obj, path):
        image = {}

        img_x = utils.img_b64_to_arr(obj['imageData'])
        image['height'], image['width'] = img_x.shape[:-1]

        self.img_id = int(os.path.basename(path).split(".json")[0])
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")

        return image

    def _annotation(self, bboxes_list, keypoints_list, json_path):
        if len(keypoints_list) != args.join_num * len(bboxes_list):
            print('you loss {} keypoint(s) with file {}'.format(args.join_num * len(bboxes_list) - len(keypoints_list), json_path))
            print('Please check ！！！')
            sys.exit()
        i = 0
        for object in bboxes_list:
            annotation = {}
            keypoints = []
            num_keypoints = 0

            label = object['label']
            bbox = object['points']
            annotation['id'] = self.ann_id
            annotation['image_id'] = self.img_id
            annotation['category_id'] = int(self.classname_to_id[label])
            annotation['iscrowd'] = 0
            annotation['area'] = 1.0
            annotation['segmentation'] = [np.asarray(bbox).flatten().tolist()]
            annotation['bbox'] = self._get_box(bbox)

            for keypoint in keypoints_list[i * args.join_num: (i + 1) * args.join_num]:
                point = keypoint['points']
                annotation['keypoints'], num_keypoints = self._get_keypoints(point[0], keypoints, num_keypoints)
            annotation['num_keypoints'] = num_keypoints

            i += 1
            self.ann_id += 1
            self.annotations.append(annotation)

    def _init_categories(self):
        for name, id in self.classname_to_id.items():
            category = {}

            category['supercategory'] = name
            category['id'] = id
            category['name'] = name
            # category['keypoint'] = ['0', '1', '2', '3', '4', '5']
            category['keypoint'] = [str(i + 1) for i in range(args.join_num)]

            self.categories.append(category)

    def to_coco(self, json_path_list):
        self._init_categories()

        for json_path in tqdm(json_path_list):
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']

            bboxes_list, keypoints_list = [], []
            for shape in shapes:
                if shape['shape_type'] == 'rectangle':
                    bboxes_list.append(shape)
                elif shape['shape_type'] == 'point':
                    keypoints_list.append(shape)

            self._annotation(bboxes_list, keypoints_list, json_path)

        keypoints = {}
        keypoints['info'] = {'description': 'Fish Pose Dataset', 'version': 1.0, 'year': 2021}
        keypoints['license'] = ['ABin']
        keypoints['images'] = self.images
        keypoints['annotations'] = self.annotations
        keypoints['categories'] = self.categories
        return keypoints


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", "--n", default='fish', help="class name", type=str)
    parser.add_argument("--input", "--i", default=r'.\data\fish', help="json file path (labelme)", type=str)
    parser.add_argument("--output", "--o", default=r'.\data', help="output file path (coco format)", type=str)
    parser.add_argument("--join_num", "--j", default=10, help="number of join", type=int)
    parser.add_argument("--ratio", "--r", default=0.1, help="train and test split ratio", type=float)
    args = parser.parse_args()

    labelme_path = args.input
    saved_coco_path = args.output

    if not os.path.exists("%scoco/annotations/"%saved_coco_path):
        os.makedirs("%scoco/annotations/"%saved_coco_path)
    if not os.path.exists("%scoco/train/"%saved_coco_path):
        os.makedirs("%scoco/train"%saved_coco_path)
    if not os.path.exists("%scoco/val/"%saved_coco_path):
        os.makedirs("%scoco/val"%saved_coco_path)

    json_list_path = glob.glob(labelme_path + "/*.json")
    # print(json_list_path)
    train_path, val_path = train_test_split(json_list_path, test_size=args.ratio)
    print('{} for training'.format(len(train_path)),
          '\n{} for testing'.format(len(val_path)))
    print('Start transform please wait ...')

    l2c_train = Labelme2coco(args)
    train_keypoints = l2c_train.to_coco(train_path)

    l2c_train.save_coco_json(train_keypoints, '%scoco/annotations/keypoints_train.json' % saved_coco_path)
    for file in train_path:
        shutil.copy(file.replace("json", "jpg"), "%scoco/train/" % saved_coco_path)
    for file in val_path:
        shutil.copy(file.replace("json", "jpg"), "%scoco/val/" % saved_coco_path)
    #
    l2c_val = Labelme2coco(args)
    val_instance = l2c_val.to_coco(val_path)
    l2c_val.save_coco_json(val_instance, '%scoco/annotations/keypoints_val.json' % saved_coco_path)

