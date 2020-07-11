#-*-coding:utf-8-*-
import sys
sys.path.append('.')
import numpy as np

import cv2
import os



from lib.core.api import Detector
data_dir='../global-wheat-detection/train'


model_path='./models/epoch_8_val_loss0.698121.pth'


image_list=os.listdir(data_dir)

detector=Detector(model_path)
for pic in image_list:
    cur_path=os.path.join(data_dir,pic)

    image=cv2.imread(cur_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image=cv2.resize(image,(640,640))

    out=detector(image,640)

    for i in range(out.shape[0]):
        cur_box=out[i]
        xmin = int(cur_box[1])
        ymin = int(cur_box[0])
        xmax = int(cur_box[3])
        ymax = int(cur_box[2])

        if cur_box[4] > 0.3:

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)


    cv2.imshow('res',image)
    cv2.waitKey(0)