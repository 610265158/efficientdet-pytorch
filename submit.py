#-*-coding:utf-8-*-
import sys
sys.path.append('.')
import numpy as np

import cv2
import os
import pandas as pd


from lib.core.api import Detector
data_dir='../global-wheat-detection/test'


model_path='./epoch_287_val_loss1.227091.pth'


image_list=os.listdir(data_dir)
image_list=[x for x in image_list if 'jpg' in x]
detector=Detector(model_path)

results=[]

score_thres=0.05
show_flag=True

def format_prediction_string(boxes):
    pred_strings = []
    for j in range(boxes.shape[0]):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(boxes[j][4],int(boxes[j][0]), int(boxes[j][1]), int(boxes[j][2]-boxes[j][0]), int(boxes[j][3]-boxes[j][1])))
    return " ".join(pred_strings)


for pic in image_list:

    cur_path=os.path.join(data_dir,pic)

    print(cur_path)
    image=cv2.imread(cur_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    out_boxes=detector(image,640,1024,iou_thres=0.5,score_thres=0.05)

    if show_flag:
        for i in range(out_boxes.shape[0]):
            cur_box=out_boxes[i]
            xmin = int(cur_box[0])
            ymin = int(cur_box[1])
            xmax = int(cur_box[2])
            ymax = int(cur_box[3])

            if cur_box[4] > 0.3:

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

        cv2.namedWindow('res',0)
        cv2.imshow('res',image)
        cv2.waitKey(0)
    cur_image_id=pic.split('.')[0]

    cur_detect_result=format_prediction_string(out_boxes)

    cur_result = {
        'image_id': cur_image_id,
        'PredictionString': cur_detect_result
    }
    results.append(cur_result)


test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('submission.csv', index=False)
test_df.head()