import numpy as np
import pandas as pd
import os
import random
import json

from sklearn.model_selection import StratifiedKFold

data_dir='../global-wheat-detection/train'
train_csv='../global-wheat-detection/train.csv'
fold_used=0


marking = pd.read_csv(train_csv)

bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
for i, column in enumerate(['x', 'y', 'w', 'h']):
    marking[column] = bboxs[:,i]
marking.drop(columns=['bbox'], inplace=True)
print(marking)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

df_folds = marking[['image_id']].copy()
df_folds.loc[:, 'bbox_count'] = 1
df_folds = df_folds.groupby('image_id').count()
df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
df_folds.loc[:, 'stratify_group'] = np.char.add(
    df_folds['source'].values.astype(str),
    df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
)
df_folds.loc[:, 'fold'] = 0

for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
    df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

print(df_folds)




train_list=df_folds[df_folds['fold'] != fold_used].index.values
val_list=df_folds[df_folds['fold'] == fold_used].index.values



train_data=pd.read_csv(train_csv)

print(train_data)


total_list_from_image=os.listdir(data_dir)
total_list_from_image=[ x.split('.')[0] for x in total_list_from_image]
#
#
image_ids=list(set(train_data['image_id']))


#
emplty_image_list=[ x for x in total_list_from_image if x not in image_ids]
#
print('the image has no label',emplty_image_list)


klasses=set(train_data['source'])

train_file=open('train.txt', 'w')



train_data_coco = {}
train_data_coco['licenses'] = []
train_data_coco['info'] = []
train_data_coco['categories'] = [{'id': 1, 'name': 'wheat', 'supercategory': 'wheat'}]
train_data_coco['images'] = []
train_data_coco['annotations'] = []

img_id=0
anno_id=0

for k,id in enumerate(train_list):

    file_name=data_dir+'/'+id+'.jpg'
    img_entry = {'file_name': file_name, 'id': img_id, 'height': 1024, 'width': 1024}
    train_data_coco['images'].append(img_entry)




    bboxes=train_data[train_data['image_id']==id]

    cur_label_message=data_dir+'/'+str(id)+'.jpg|'

    for box in bboxes['bbox']:
        curbox=box[1:-1].split(',')
        cur_box_info=[float(x) for x in curbox]
        xmin = int(cur_box_info[0])
        ymin = int(cur_box_info[1])
        xmax = int(cur_box_info[0]+cur_box_info[2])
        ymax = int(cur_box_info[1]+cur_box_info[3])

        anno_entry = {'image_id': img_id, 'category_id': 1, 'id': anno_id,\
                        'iscrowd': 0, 'area': int(xmax-xmin) * int(ymax-ymin),\
                        'bbox': [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)]}
        train_data_coco['annotations'].append(anno_entry)

        anno_id+=1
    img_id+=1
with open('./Train_cocoStyle.json', 'w') as outfile:
    json.dump(train_data_coco, outfile,indent=2)


test_data_coco = {}
test_data_coco['licenses'] = []
test_data_coco['info'] = []
test_data_coco['categories'] = [{'id': 1, 'name': 'wheat', 'supercategory': 'wheat'}]
test_data_coco['images'] = []
test_data_coco['annotations'] = []

for k,id in enumerate(val_list):



    file_name = data_dir + '/' + id + '.jpg'
    img_entry = {'file_name': file_name, 'id': img_id, 'height': 1024, 'width': 1024}
    test_data_coco['images'].append(img_entry)


    bboxes=train_data[train_data['image_id']==id]

    for box in bboxes['bbox']:
        curbox = box[1:-1].split(',')
        cur_box_info = [float(x) for x in curbox]
        xmin = int(cur_box_info[0])
        ymin = int(cur_box_info[1])
        xmax = int(cur_box_info[0] + cur_box_info[2])
        ymax = int(cur_box_info[1] + cur_box_info[3])

        anno_entry = {'image_id': img_id, 'category_id': 1, 'id': anno_id, \
                      'iscrowd': 0, 'area': int(xmax - xmin) * int(ymax - ymin), \
                      'bbox': [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]}
        test_data_coco['annotations'].append(anno_entry)

        anno_id += 1
    img_id += 1


with open('./Val_cocoStyle.json', 'w') as outfile:
    json.dump(test_data_coco, outfile,indent=2)












