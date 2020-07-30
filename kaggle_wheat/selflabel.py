import argparse
import sys
sys.path.append('.')

import cv2
import numpy as np
from tqdm import tqdm

from lib.core.utils import calculate_image_precision, iou_thresholds
from lib.dataset.augmentor.augmentation import Fill_img
from lib.dataset.dataietr import DataIter, data_info
from lib.utils.logger import logger
from lib.wbf.ensemble_boxes import weighted_boxes_fusion, non_maximum_weighted, soft_nms,nms

from lib.core.api import Detector
from lib.dataset.augmentor.augmentation import Rotate_aug

from train_config import config as cfg
import os

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=False, default='./model/detector.pb', help="model to eval:")

ap.add_argument("--is_show", required=False, default=0,type=int, help="show result or not?")
args = ap.parse_args()



argmodel_name=args.model
model_names=[argmodel_name]

is_show=args.is_show

data_dir='../global-wheat-detection/test/'
selflabel_dir='./selflabel'
if not os.access(selflabel_dir,os.F_OK):
    os.mkdir(selflabel_dir)



image_list=os.listdir(data_dir)


def process_det(det, score_threshold=0.25):

    indexes = (det[:,4]>score_threshold)
    boxes = det[indexes,:4]
    scores = det[indexes,4]
    return boxes, scores

def get_prediction(angle=15):
    all_predictions = {}

    for model_name in model_names:

        detector = Detector(model_name)

        for k,image_name in enumerate(image_list):

            cur_image_path=os.path.join(data_dir,image_name)

            image=cv2.imread(cur_image_path,-1)

            image,_=Rotate_aug(image,angle)


            ###save there
            save_name=os.path.join(selflabel_dir,str(angle)+'_'+image_name)
            cv2.imwrite(save_name,image)


            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            image_show=np.array(image)

            cur_result = detector(image,512)

            if image_name in all_predictions:
                pass
            else:
                all_predictions[image_name]={}

            pred_boxes,pred_score=process_det(cur_result)

            all_predictions[image_name]['pred_boxes_with_model_%s' % (model_name)]=pred_boxes
            all_predictions[image_name]['pred_scores_with_model%s' % (model_name)] = pred_score

            if is_show:
                for i in range(cur_result.shape[0]):
                    one_box = cur_result[i]
                    str_draw = str(int(one_box[5])) + ' score:' + str(one_box[4])
                    if one_box[4] > 0.3:
                        cv2.rectangle(image_show, (int(one_box[0]), int(one_box[1])),
                                      (int(one_box[2]), int(one_box[3])),
                                      (0, 255, 0), 2)
                        cv2.putText(image_show, str_draw, (int(one_box[0]), int(one_box[1])), cv2.FONT_HERSHEY_SIMPLEX,
                                    2,
                                    (0, 255, 0), 3)

                # cv2.imwrite('./tmp/'+str(img_id)+'.jpg',image_show)
                cv2.namedWindow('ss', 0)
                cv2.imshow('ss', image_show)
                cv2.waitKey(0)



    return all_predictions

def calculate_final_score(
        all_predictions,
        iou_thr,
        skip_box_thr,
        method,  # weighted_boxes_fusion, nms, soft_nms, non_maximum_weighted
        sigma=0.5,
        angle=15
):


    selflabel_file= open('selflabel.txt','a')

    for k,v in all_predictions.items():


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


        image_name = k
        cur_label_message='nan| '+os.path.join(selflabel_dir,str(angle)+'_'+image_name+'|')
        ####make the txt
        for bb in range(boxes.shape[0]):

            curbox = boxes[bb]
            cur_box_info = [float(x) for x in curbox]

            cur_box_info = " " + str(cur_box_info[0]) + ',' + str(cur_box_info[1]) + ',' + \
                           str( cur_box_info[2]) + "," + str( cur_box_info[3]) + ',1'
            cur_label_message = cur_label_message + cur_box_info

        cur_label_message += '\n'
        selflabel_file.write(cur_label_message)



if __name__=='__main__':

    for angle in [15,30,45,60,75]:
        all_predictions=get_prediction(angle=angle)



        score=calculate_final_score(all_predictions,0.5,0.430,'weighted_boxes_fusion',angle=angle)


