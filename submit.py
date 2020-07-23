#-*-coding:utf-8-*-
import sys

sys.path.append('.')
import numpy as np

import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt

from lib.core.api import Detector
from lib.wbf.ensemble_boxes import weighted_boxes_fusion, non_maximum_weighted, soft_nms,nms

data_dir='../global-wheat-detection/test'

model_names=['epoch_135_val_loss0.391522.pth']


image_list=os.listdir(data_dir)
image_list=[x for x in image_list if 'jpg' in x]


results=[]
iou_thres=0.430
score_thres=0.430
is_show=True

def format_prediction_string(boxes):
    pred_strings = []
    for j in range(boxes.shape[0]):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(boxes[j][4],int(boxes[j][0]), int(boxes[j][1]), int(boxes[j][2]-boxes[j][0]), int(boxes[j][3]-boxes[j][1])))
    return " ".join(pred_strings)



def process_det(det, score_threshold=0.25):

    indexes = (det[:,4]>score_threshold)
    boxes = det[indexes,:4]
    scores = det[indexes,4]
    return boxes, scores

def get_prediction():
    all_predictions={}
    for model_name in model_names:

        detector = Detector(model_name)

        for pic in image_list:

            cur_path = os.path.join(data_dir, pic)

            print(cur_path)
            image = cv2.imread(cur_path)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            cur_result = detector(image)

            image_id=pic.split('.')[0]

            if image_id in all_predictions:
                pass
            else:
                all_predictions[image_id]={}


            if 'gt_boxes' in all_predictions[image_id]:
                pass
            else:

                pred_boxes,pred_score=process_det(cur_result)

                all_predictions[image_id]['pred_boxes_with_model_%s' % (model_name)]=pred_boxes
                all_predictions[image_id]['pred_scores_with_model%s' % (model_name)] = pred_score

            if is_show:
                for i in range(cur_result.shape[0]):
                    one_box = cur_result[i]
                    str_draw = str(int(one_box[5])) + ' score:' + str(one_box[4])
                    if one_box[4] > 0.3:
                        cv2.rectangle(image, (int(one_box[0]), int(one_box[1])),
                                      (int(one_box[2]), int(one_box[3])),
                                      (0, 255, 0), 2)
                        cv2.putText(image, str_draw, (int(one_box[0]), int(one_box[1])), cv2.FONT_HERSHEY_SIMPLEX,
                                    2,
                                    (0, 255, 0), 3)

                # cv2.imwrite('./tmp/'+str(img_id)+'.jpg',image_show)
                cv2.namedWindow('ss', 0)
                cv2.imshow('ss', image)
                cv2.waitKey(0)

    return all_predictions

def ensemble(all_predictions,
        iou_thr,
        skip_box_thr,
        method,  # weighted_boxes_fusion, nms, soft_nms, non_maximum_weighted
        sigma=0.5,):

    results=[]
    for k,v in all_predictions.items():

        cur_image_id = k
        folds_boxes, folds_scores, folds_labels = [], [], []
        for model_name in model_names:
            pred_boxes = all_predictions[k]['pred_boxes_with_model_%s' % (model_name)].copy()
            scores = all_predictions[k]['pred_scores_with_model%s' % (model_name)].copy()
            folds_boxes.append(pred_boxes)
            folds_scores.append(scores)
            folds_labels.append(np.ones(pred_boxes.shape[0]))

        if method == 'weighted_boxes_fusion':
            boxes, scores, labels = weighted_boxes_fusion(folds_boxes, folds_scores, folds_labels, weights=None,
                                                          iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        elif method == 'nms':
            boxes, scores, labels = nms(folds_boxes, folds_scores, folds_labels, weights=None, iou_thr=iou_thr)
        elif method == 'soft_nms':
            boxes, scores, labels = soft_nms(folds_boxes, folds_scores, folds_labels, weights=None, iou_thr=iou_thr,
                                             thresh=skip_box_thr, sigma=sigma)
        elif method == 'non_maximum_weighted':
            boxes, scores, labels = non_maximum_weighted(folds_boxes, folds_scores, folds_labels, weights=None,
                                                         iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        else:
            raise NotImplementedError


        scores=np.expand_dims(scores,axis=-1)
        labels=np.expand_dims(labels,axis=-1)

        cur_detect_result=np.concatenate([boxes,scores,labels],axis=1)

        cur_result = {
                'image_id': cur_image_id,
                'PredictionString': format_prediction_string(cur_detect_result)
            }
        results.append(cur_result)
    return results

all_predictions=get_prediction()

results=ensemble(all_predictions,iou_thr=0.430,skip_box_thr=0.430,method='weighted_boxes_fusion')

test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.to_csv('../submission.csv', index=False)
test_df.head(10)