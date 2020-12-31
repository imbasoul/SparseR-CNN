import os
import numpy as np
import cv2
from tqdm import tqdm
from detectron2.structures import BoxMode


def get_face_dicts(img_dir, eval=False):
    # json_file = os.path.join(img_dir, "annotations")
    img_path = os.path.join(img_dir, "images")
    anno_path = os.path.join(img_dir, "annotations")
    dataset_dicts = []
    for i, files in enumerate(tqdm(os.listdir(anno_path))):  # 不仅仅是文件，当前目录下的文件夹也会被认为遍历到
        if eval==False and (i<=4000 or i>=3000):
            continue
        elif eval==True and (i>4000 or i<3000):
            continue
        record = {}
        # with open(files) as f:
        anno = np.loadtxt(os.path.join(anno_path, files), usecols=(1, 2, 3, 4))
        if eval==True and len(anno)==0:
            continue
        single_img_path = os.path.join(img_path, files[:-4]+".jpg")
        # print("img_path", single_img_path)
        try:
            height, width = cv2.imread(single_img_path).shape[:2]
            if not height:
                continue
            if not width:
                continue
        except:
            print("imread_error", single_img_path)
            continue
        record["file_name"] = single_img_path
        record["image_id"] = files
        record["height"] = height
        record["width"] = width
        objs = []
        if isinstance(anno[0], np.float64):
            anno = [anno]
        for i in range(len(anno)):
            obj = {
                "bbox": [anno[i][0],anno[i][1],anno[i][2],anno[i][3]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,
            }
            objs.append(obj)
        # print(objs)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def get_face_test_dicts(img_dir):
    # json_file = os.path.join(img_dir, "annotations")
    img_path = os.path.join(img_dir, "images")
    # anno_path = os.path.join(img_dir, "annotations")
    dataset_dicts = []
    for files in tqdm(os.listdir(img_path)):  # 不仅仅是文件，当前目录下的文件夹也会被认为遍历到
        record = {}
        # with open(files) as f:
        # anno = np.loadtxt(os.path.join(anno_path, files), usecols=(1, 2, 3, 4))
        single_img_path = os.path.join(img_path, files)
        # print("img_path", single_img_path)
        height, width = cv2.imread(single_img_path).shape[:2]
        record["file_name"] = single_img_path
        record["image_id"] = files
        record["height"] = height
        record["width"] = width
        objs = []
        # if isinstance(anno[0], np.float64):
        #     anno = [anno]
        # for i in range(len(anno)):
        #     obj = {
        #         "bbox": [anno[i][0], anno[i][1], anno[i][2], anno[i][3]],
        #         "bbox_mode": BoxMode.XYXY_ABS,
        #         "category_id": 0,
        #     }
        #     objs.append(obj)
        # print(objs)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
        # print("files", files)

    # dataset_dicts = []
    # for idx, v in enumerate(imgs_anns.values()):
    #     record = {}
    # 
    #     filename = os.path.join(img_dir, v["filename"])
    #     height, width = cv2.imread(filename).shape[:2]
    # 
    #     record["file_name"] = filename
    #     record["image_id"] = idx
    #     record["height"] = height
    #     record["width"] = width
    # 
    #     annos = v["regions"]
    #     objs = []
    #     for _, anno in annos.items():
    #         assert not anno["region_attributes"]
    #         anno = anno["shape_attributes"]
    #         px = anno["all_points_x"]
    #         py = anno["all_points_y"]
    #         poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
    #         poly = [p for x in poly for p in x]
    # 
    #         obj = {
    #             "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
    #             "bbox_mode": BoxMode.XYXY_ABS,
    #             "segmentation": [poly],
    #             "category_id": 0,
    #         }
    #         objs.append(obj)
    #     record["annotations"] = objs
    #     dataset_dicts.append(record)
    # return dataset_dicts


if __name__ == '__main__':
    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
    balloon_metadata = MetadataCatalog.get("balloon_train")