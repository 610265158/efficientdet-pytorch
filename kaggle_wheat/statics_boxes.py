import pandas as pd
import numpy as np

data_dir='../global-wheat-detection/train'
train_csv='../global-wheat-detection/train.csv'
fold_used=0


marking = pd.read_csv(train_csv)



bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))





box_dict={'small':0,
          'media':0,
          'large':0,
          'except':0,

          '2/10':0,
          '10/2':0,
          'other':0}


for bb in bboxs:
    area=bb[2]*bb[3]

    if area<=32*32:
        box_dict['small']+=1
    elif area>32*32 and area<=96*96:
        box_dict['media'] += 1
    elif area > 96*96 and area <= 512*512:
        box_dict['large'] += 1
    else:
        box_dict['except'] += 1


    if bb[2]/bb[3]<=0.2:
        box_dict['2/10'] += 1
    elif bb[3]/bb[1]<=0.2:
        box_dict['10/2'] += 1
    else:
        box_dict['other'] += 1
for k,v in box_dict.items():
    print(k,v)