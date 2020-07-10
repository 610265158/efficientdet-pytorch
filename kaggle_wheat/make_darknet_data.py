
import numpy as np
import pandas as pd
import json
import os
import shutil
data_wheat='../yolo_data_wheat'

images_dir=os.path.join(data_wheat,'images')
labels_dir=os.path.join(data_wheat,'labels')
train_images_dir=os.path.join(images_dir,'train')
val_images_dir=os.path.join(images_dir,'val')
train_labels_dir=os.path.join(labels_dir,'train')
val_labels_dir=os.path.join(labels_dir,'val')
if not os.access(data_wheat,os.F_OK):
    os.mkdir(data_wheat)
    os.mkdir(images_dir)
    os.mkdir(labels_dir)
    os.mkdir(train_images_dir)
    os.mkdir(val_images_dir)
    os.mkdir(train_labels_dir)
    os.mkdir(val_labels_dir)


data_dir='../global-wheat-detection/train'
train_csv='../global-wheat-detection/train.csv'


ratio=0.9

train_data=pd.read_csv(train_csv)

print(train_data)


image_ids=list(set(train_data['image_id']))

train_list=image_ids[:int(ratio*len(image_ids))]
val_list=image_ids[int(ratio*len(image_ids)):]

klasses=set(train_data['source'])




img_id=0
anno_id=0


for k,id in enumerate(train_list):

    file_name=data_dir+'/'+id+'.jpg'

    ###raw dataset
    to_path=os.path.join(train_images_dir,id+'_raw.jpg')
    shutil.copy(file_name,to_path)

    bboxes=train_data[train_data['image_id']==id]

    label_text_file=os.path.join(train_labels_dir,id+'_raw.txt')
    with open(label_text_file,'w') as f:

        for box in bboxes['bbox']:
            curbox=box[1:-1].split(',')
            cur_box_info=[float(x) for x in curbox]
            x_center = (cur_box_info[0]+cur_box_info[2]/2)/1024
            y_center = (cur_box_info[1]+cur_box_info[3]/2)/1024
            width    = (cur_box_info[2])/1024
            height   = (cur_box_info[3])/1024
            cur_line_message='0 '+str(x_center)+' '+str(y_center)+' '+str(width)+' '+str(height)+'\n'
            f.write(cur_line_message)



for k,id in enumerate(val_list):

    file_name = data_dir + '/' + id + '.jpg'

    ###raw dataset
    to_path = os.path.join(val_images_dir, id + '_raw.jpg')
    shutil.copy(file_name, to_path)

    bboxes = train_data[train_data['image_id'] == id]

    label_text_file = os.path.join(val_labels_dir, id + '_raw.txt')
    with open(label_text_file,'w') as f:

        for box in bboxes['bbox']:
            curbox = box[1:-1].split(',')
            cur_box_info = [float(x) for x in curbox]
            x_center = (cur_box_info[0] + cur_box_info[2] / 2) / 1024
            y_center = (cur_box_info[1] + cur_box_info[3] / 2) / 1024
            width = (cur_box_info[2]) / 1024
            height = (cur_box_info[3]) / 1024
            cur_line_message = '0 ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n'
            f.write(cur_line_message)













