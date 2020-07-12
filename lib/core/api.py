#-*-coding:utf-8-*-
import numpy as np
import torch

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain,DetBenchPredict
from effdet.efficientdet import HeadNet

from train_config import config as cfg
import cv2




class Detector():
    def __init__(self,model_path):


        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        config = get_efficientdet_config('tf_efficientdet_d2')
        net = EfficientDet(config, pretrained_backbone=False)

        config.num_classes = 1
        config.image_size = cfg.DATA.hin

        net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

        state_dict = torch.load(model_path, map_location=self.device)
        net.load_state_dict(state_dict, strict=False)


        self.model = DetBenchPredict(net, config)

        self.model = self.model.cuda()

        self.model.eval()


        self.mean = torch.tensor([x * 255 for x in cfg.DATA.IMAGENET_DEFAULT_MEAN]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor([x * 255 for x in cfg.DATA.IMAGENET_DEFAULT_MEAN]).cuda().view(1, 3, 1, 1)


    def __call__(self, img, input_size=640,raw_image_size=1024,iou_thres=0.5,score_thres=0.05):

        img = cv2.resize(img, (input_size, input_size))
        img=np.transpose(img,axes=[2,0,1])
        img=np.expand_dims(img,0)
        data = torch.from_numpy(img).to(self.device).float()

        data = data.float().sub_(self.mean).div_(self.std)


        with torch.no_grad():
            output=self.model(data,input_size)

        output=output.cpu().numpy()[0]


        result=self.py_nms(output,iou_thres=iou_thres,score_thres=score_thres,max_boxes=2000)


        ##yxyx to xyxy

        result=result[:,[1,0,3,2,4,5]]
        result[:, 0:4] = result[:, 0:4] / input_size * raw_image_size


        return result

    def py_nms(self, bboxes, iou_thres, score_thres, max_boxes=1000):

        upper_thres = np.where(bboxes[:, 4] > score_thres)[0]

        bboxes = bboxes[upper_thres]
        if iou_thres is None:
            return bboxes

        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        order = np.argsort(bboxes[:, 4])[::-1]

        keep = []
        while order.shape[0] > 0:
            if len(keep) > max_boxes:
                break
            cur = order[0]

            keep.append(cur)

            area = (bboxes[cur, 2] - bboxes[cur, 0]) * (bboxes[cur, 3] - bboxes[cur, 1])

            x1_reain = x1[order[1:]]
            y1_reain = y1[order[1:]]
            x2_reain = x2[order[1:]]
            y2_reain = y2[order[1:]]

            xx1 = np.maximum(bboxes[cur, 0], x1_reain)
            yy1 = np.maximum(bboxes[cur, 1], y1_reain)
            xx2 = np.minimum(bboxes[cur, 2], x2_reain)
            yy2 = np.minimum(bboxes[cur, 3], y2_reain)

            intersection = np.maximum(0, yy2 - yy1) * np.maximum(0, xx2 - xx1)

            iou = intersection / (area + (y2_reain - y1_reain) * (x2_reain - x1_reain) - intersection)

            ##keep the low iou
            low_iou_position = np.where(iou < iou_thres)[0]

            order = order[low_iou_position + 1]

        return bboxes[keep]

    def load_weight(self,path):

        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)