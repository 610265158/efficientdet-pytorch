#-*-coding:utf-8-*-

import os
import numpy
import cv2

raw_img_dir='../global-wheat-detection/train'

low_resolution_dir='../global-wheat-detection/low_resolution'

if not os.access(low_resolution_dir,os.F_OK):
    os.mkdir(low_resolution_dir)

for pic in os.listdir(raw_img_dir):
    cur_path=os.path.join(raw_img_dir,pic)


    img=cv2.imread(cur_path)

    img=cv2.blur(img,ksize=(3,3))

    file_name=os.path.join(low_resolution_dir,pic.replace('.jpg','_3x3.jpg'))
    cv2.imwrite(file_name,img)





