import numpy as np
import pandas as pd
import os
import random

from sklearn.model_selection import StratifiedKFold

data_dir='../global-wheat-detection/train'
train_csv='../global-wheat-detection/train.csv'


fold_used=0


extern_data=True
if extern_data:
    extern_dir='../global-wheat-detection/adv_data_all'
    extern_image_list=os.listdir(extern_dir)
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



''''

some bad bbox
3687
117344
173
113947
52868
2159
2169
121633
121634
147504
118211
52727
147552
'''
bad_box_index=[3687,117344,173,113947,52868,2159,2169,121633,121634,147504,118211,52727,147552]


###drop the bad box
for drop_id in bad_box_index:
    train_data=train_data.drop(index=drop_id)


for k,id in enumerate(train_list):

    bboxes=train_data[train_data['image_id']==id]

    cur_label_message=data_dir+'/'+str(id)+'.jpg|'

    for box in bboxes['bbox']:
        curbox=box[1:-1].split(',')
        cur_box_info=[float(x) for x in curbox]

        cur_box_info=" "+ str(cur_box_info[0])+',' + str(cur_box_info[1])+ ','+\
                     str(cur_box_info[0]+cur_box_info[2])+","+str(cur_box_info[1]+cur_box_info[3]) +',1'
        cur_label_message=cur_label_message+cur_box_info

    cur_label_message+='\n'
    train_file.write(cur_label_message)



###write the empty image with train  label as 0,0,0,0,0

for k,id in enumerate(emplty_image_list):


    cur_label_message=data_dir+'/'+str(id)+'.jpg| 0,0,0,0,0'

    cur_label_message+='\n'
    train_file.write(cur_label_message)

if extern_data:
    for k, img_name in enumerate(extern_image_list):
        cur_label_message = extern_dir + '/' + str(img_name) + '| 0,0,0,0,0'

        cur_label_message += '\n'
        train_file.write(cur_label_message)
        
val_file=open('val.txt', 'w')


for k,id in enumerate(val_list):

    bboxes=train_data[train_data['image_id']==id]

    cur_label_message=data_dir+'/'+str(id)+'.jpg|'

    for box in bboxes['bbox']:
        curbox=box[1:-1].split(',')
        cur_box_info=[float(x) for x in curbox]

        cur_box_info=" "+ str(cur_box_info[0])+',' + str(cur_box_info[1])+ ','+\
                     str(cur_box_info[0]+cur_box_info[2])+","+str(cur_box_info[1]+cur_box_info[3]) +',1'
        cur_label_message=cur_label_message+cur_box_info

    cur_label_message+='\n'
    val_file.write(cur_label_message)


