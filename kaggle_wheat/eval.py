
import sys

sys.path.append('.')

import cv2
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from train_config import config as cfg
from lib.core.api import Detector



ap = argparse.ArgumentParser()
ap.add_argument("--model", required=False, default='./model/detector.pb', help="model to eval:")
ap.add_argument("--rootpath", required=False, default='', help="model to eval:")
ap.add_argument("--annFile", required=False, default='./Val_cocoStyle.json', help="coco style json")
ap.add_argument("--is_show", required=False, default=0,type=int, help="show result or not?")
args = ap.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

coco_map = {0: (1, 'person'), 1: (2, 'bicycle'), 2: (3, 'car'), 3: (4, 'motorcycle'), 4: (5, 'airplane'), 5: (6, 'bus'),
            6: (7, 'train'), 7: (8, 'truck'), 8: (9, 'boat'), 9: (10, 'traffic light'), 10: (11, 'fire hydrant'),
            11: (13, 'stop sign'), 12: (14, 'parking meter'), 13: (15, 'bench'), 14: (16, 'bird'), 15: (17, 'cat'),
            16: (18, 'dog'), 17: (19, 'horse'), 18: (20, 'sheep'), 19: (21, 'cow'), 20: (22, 'elephant'),
            21: (23, 'bear'), 22: (24, 'zebra'), 23: (25, 'giraffe'), 24: (27, 'backpack'), 25: (28, 'umbrella'),
            26: (31, 'handbag'), 27: (32, 'tie'), 28: (33, 'suitcase'), 29: (34, 'frisbee'), 30: (35, 'skis'),
            31: (36, 'snowboard'), 32: (37, 'sports ball'), 33: (38, 'kite'), 34: (39, 'baseball bat'),
            35: (40, 'baseball glove'),
            36: (41, 'skateboard'), 37: (42, 'surfboard'), 38: (43, 'tennis racket'), 39: (44, 'bottle'),
            40: (46, 'wine glass'),
            41: (47, 'cup'), 42: (48, 'fork'), 43: (49, 'knife'), 44: (50, 'spoon'), 45: (51, 'bowl'),
            46: (52, 'banana'), 47: (53, 'apple'), 48: (54, 'sandwich'), 49: (55, 'orange'), 50: (56, 'broccoli'),
            51: (57, 'carrot'), 52: (58, 'hot dog'), 53: (59, 'pizza'), 54: (60, 'donut'), 55: (61, 'cake'),
            56: (62, 'chair'), 57: (63, 'couch'), 58: (64, 'potted plant'), 59: (65, 'bed'), 60: (67, 'dining table'),
            61: (70, 'toilet'), 62: (72, 'tv'), 63: (73, 'laptop'), 64: (74, 'mouse'), 65: (75, 'remote'),
            66: (76, 'keyboard'), 67: (77, 'cell phone'), 68: (78, 'microwave'), 69: (79, 'oven'), 70: (80, 'toaster'),
            71: (81, 'sink'), 72: (82, 'refrigerator'), 73: (84, 'book'), 74: (85, 'clock'), 75: (86, 'vase'),
            76: (87, 'scissors'), 77: (88, 'teddy bear'), 78: (89, 'hair drier'), 79: (90, 'toothbrush')}


def predict_box():
    ROOT_PATH= args.rootpath
    MODEL_PATH = args.model
    detector = Detector(MODEL_PATH)
    annFile = args.annFile
    cocoGt = COCO(annFile)
    catIds = cocoGt.getCatIds()
    print(catIds)

    imgIds = sorted(cocoGt.getImgIds())
    print(imgIds)
    res_coco = []

    for img_id in tqdm(imgIds):


        fname=cocoGt.loadImgs(img_id)[0]['file_name']

        fname=os.path.join(ROOT_PATH,fname)


        image = cv2.imread(fname)
        image_show = image.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w, _ = image.shape


        detect_res =detector(image,768,iou_thres=0.5,score_thres=0.03)



        ##recover to raw size



        if args.is_show:
            for i in range(detect_res.shape[0]):
                one_box = detect_res[i]
                str_draw =  str(int(one_box[5]))+' score:' + str(one_box[4])
                if one_box[4]>0.3:
                    cv2.rectangle(image_show, (int(one_box[0]), int(one_box[1])), (int(one_box[2]), int(one_box[3])),
                                  (0, 255, 0), 2)
                    cv2.putText(image_show, str_draw, (int(one_box[0]), int(one_box[1])), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (0, 255, 0), 3)


            # cv2.imwrite('./tmp/'+str(img_id)+'.jpg',image_show)
            cv2.namedWindow('ss',0)
            cv2.imshow('ss', image_show)
            cv2.waitKey(0)

        for i in range(detect_res.shape[0]):
            one_box = detect_res[i]

            if one_box[4] < .001:  # stop when below this threshold, scores in descending order
                continue
            one_box=[float(x) for x in one_box]
            box = [one_box[0], one_box[1], one_box[2] - one_box[0], one_box[3] - one_box[1]]
            res_coco.append({
                'bbox': box,
                'category_id': int(one_box[5]),
                'image_id': img_id,
                'score': one_box[4]
            })

    with open('bbox_result.json', 'w') as f_dump:
        json.dump(res_coco, f_dump, indent=2)


def eval_box():

    import pylab
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)
    annType = ['segm', 'bbox', 'keypoints']
    annType = annType[1]  # specify type here
    print('Running for *%s* results.' % (annType))
    # initialize COCO ground truth api
    annFile = args.annFile
    cocoGt = COCO(annFile)
    catIds = cocoGt.getCatIds()
    imgIds = sorted(cocoGt.getImgIds(catIds=catIds))
    # initialize COCO detections api
    resFile = './bbox_result.json'
    cocoDt = cocoGt.loadRes(resFile)
    # running evaluation
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    # cocoEval.params.imgIds  = imgIds

    cocoEval.params.catIds = catIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    predict_box()
    eval_box()


