import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
data_dir='../global-wheat-detection/train'
train_csv='../global-wheat-detection/train.csv'



train_data=pd.read_csv(train_csv)

print(train_data)


image_ids=list(set(train_data['image_id']))


klasses=set(train_data['source'])
show_flag=False
for k,id in enumerate(image_ids):
    print(k)
    cur_image_path=os.path.join(data_dir,id+'.jpg')

    image=cv2.imread(cur_image_path)

    bboxes=train_data[train_data['image_id']==id]



    for index, row in bboxes.iterrows():

        box=row['bbox'][1:-1].split(',')

        box =[float(x) for x in box]
        xmin = int(box[0])
        ymin = int(box[1])
        xmax = int(box[0] + box[2])
        ymax = int(box[1] + box[3])

    if box[2]<15 or box[3]<15:
        show_flag=True

        cv2.rectangle(image,(xmin,ymin),(xmax,ymax),(0,255,0),2)

        if show_flag:
            cv2.namedWindow('ss',0)
            cv2.imshow('ss',image)
            key=cv2.waitKey(0)

            if key==ord('y'):
                print('drop ', index)
                train_data.drop(index=index)
        show_flag=False


train_data.to_csv('train_clean.csv')

