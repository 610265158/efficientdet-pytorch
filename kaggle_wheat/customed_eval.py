import argparse
import sys
sys.path.append('.')

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.core.utils import calculate_image_precision, iou_thresholds
from lib.dataset.augmentor.augmentation import Fill_img
from lib.dataset.dataietr import DataIter, data_info
from lib.utils.logger import logger
from lib.wbf.ensemble_boxes import weighted_boxes_fusion, non_maximum_weighted, soft_nms,nms

from lib.core.api import Detector


from train_config import config as cfg


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=False, default='./model/detector.pb', help="model to eval:")

ap.add_argument("--is_show", required=False, default=0,type=int, help="show result or not?")
args = ap.parse_args()



argmodel_name=args.model
model_names=[argmodel_name]

is_show=args.is_show


class ValDataIter():

    def __init__(self, img_root_path='', ann_file=None, training_flag=True, shuffle=True):



        self.training_flag = training_flag

        self.lst = self.parse_file(img_root_path, ann_file)

        self.shuffle = shuffle

        self.size=len(self.lst)
    def __getitem__(self, item):
        a,b,c=self._map_func(self.lst[item])
        return a,b,c

    def __len__(self):
        return len(self.lst)

    def parse_file(self,im_root_path,ann_file):
        '''
        :return: [fname,lbel]     type:list
        '''
        logger.info("[x] Get dataset from {}".format(im_root_path))

        ann_info = data_info(im_root_path, ann_file)
        all_samples = ann_info.get_all_sample()

        return all_samples




    def cv2_read_rgb(self,fname):
        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


    def load_image_and_boxes(self, dp):

        fname, annos = dp
        image = self.cv2_read_rgb(fname)
        labels = annos.split(' ')
        boxes = []

        for label in labels:
            bbox = np.array(label.split(','), dtype=np.float)
            if bbox[4]==0:
                ###it is fake data we will drop it
                boxes.append([0,0,1,1, bbox[4]])
            else:
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]])

        boxes = np.array(boxes, dtype=np.float)

        return image, boxes


    def _map_func(self,dp):
        """Data augmentation function."""
        ####customed here

        image,boxes=self.load_image_and_boxes(dp)

        image_id=dp[0].split('/')[-1].split('.')[0]

        return image, boxes, image_id

val_ds=ValDataIter(cfg.DATA.root_path, cfg.DATA.val_txt_path, False)

def process_det(det, score_threshold=0.25):

    indexes = (det[:,4]>score_threshold)
    boxes = det[indexes,:4]
    scores = det[indexes,4]
    return boxes, scores

def get_prediction():
    all_predictions = {}

    for model_name in model_names:

        detector = Detector(model_name)



        for step,data in enumerate(val_ds):

            images, boxes_target, image_id = data
            image_show=np.array(images)

            cur_result = detector(images)

            if image_id in all_predictions:
                pass
            else:
                all_predictions[image_id]={}


            if 'gt_boxes' in all_predictions[image_id]:
                pass
            else:

                pred_boxes,pred_score=process_det(cur_result)

                all_predictions[image_id]['gt_boxes']= (boxes_target).clip(min=0, max=1023).astype(int)
                all_predictions[image_id]['pred_boxes_with_model_%s' % (model_name)]=pred_boxes
                all_predictions[image_id]['pred_scores_with_model%s' % (model_name)] = pred_score

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
):
    final_scores = []

    for k,v in all_predictions.items():

        gt_boxes = all_predictions[k]['gt_boxes'].copy()
        image_id = k
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
        image_precision = calculate_image_precision(gt_boxes, boxes, thresholds=iou_thresholds, form='pascal_voc')
        final_scores.append(image_precision)

    return np.mean(final_scores)

if __name__=='__main__':
    all_predictions=get_prediction()


    ### from kagglenotebook, best score with 0.430 0.430
    score=calculate_final_score(all_predictions,0.430,0.430,'weighted_boxes_fusion')

    print(score)