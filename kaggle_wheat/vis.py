#-*-coding:utf-8-*-
import sys
sys.path.append('.')
import numpy as np

import cv2
import os



from lib.core.api import Detector
data_dir='../global-wheat-detection/test'


model_path='./models/epoch_99_val_loss0.410587.pth'


image_list=os.listdir(data_dir)

detector=Detector(model_path)
for pic in image_list:
    cur_path=os.path.join(data_dir,pic)

    image=cv2.imread(cur_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    out=detector(image,640,1024,iou_thres=0.5,score_thres=0.03)

    for i in range(out.shape[0]):
        cur_box=out[i]
        xmin = int(cur_box[0])
        ymin = int(cur_box[1])
        xmax = int(cur_box[2])
        ymax = int(cur_box[3])

        if cur_box[4] > 0.3:

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

    cv2.namedWindow('res',0)
    cv2.imshow('res',image)
    cv2.waitKey(0)