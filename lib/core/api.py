#-*-coding:utf-8-*-
import numpy as np
import torch

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain,DetBenchPredict
from effdet.efficientdet import HeadNet

from train_config import config as cfg
import cv2

from lib.core.wbf import weighted_boxes_fusion


class Detector():
    def __init__(self,model_path):


        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        config = get_efficientdet_config(cfg.MODEL.model_name)
        net = EfficientDet(config, pretrained_backbone=False)

        config.num_classes = 1
        config.image_size = cfg.DATA.hin

        net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

        state_dict = torch.load(model_path, map_location=self.device)
        net.load_state_dict(state_dict, strict=False)


        self.model = DetBenchPredict(net, config)

        self.model = self.model.to(self.device)

        self.model.eval()


        self.mean = torch.tensor(cfg.DATA.IMAGENET_DEFAULT_MEAN).to(self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.DATA.IMAGENET_DEFAULT_MEAN).to(self.device).view(1, 3, 1, 1)


    def __call__(self, image, input_size=640,iou_thres=0.5,score_thres=0.05):

        raw_h, raw_w, _ = image.shape

        max_edge = max(raw_h, raw_w)

        scale = input_size / max_edge

        img_resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        img = np.full(shape=[input_size, input_size, 3], fill_value=cfg.DATA.IMAGENET_DEFAULT_MEAN)

        resized_h, resized_w, _ = img_resized.shape
        img[:resized_h, :resized_w, :] = img_resized


        img=np.transpose(img,axes=[2,0,1])
        img=np.expand_dims(img,0)
        data = torch.from_numpy(img).to(self.device).float()

        data = data.float().sub_(self.mean).div_(self.std)


        with torch.no_grad():
            output=self.model(data,input_size)

        output=output.cpu().numpy()[0]

        result=self.py_nms(output,iou_thres=iou_thres,score_thres=score_thres,max_boxes=2000)

        ###scale back to raw image
        result[:, 0:4] = result[:, 0:4] / scale

        ##clip
        result[:, 0:2][result[:, 0:2] < 0] = 0

        result[:, 3][result[:, 3] > raw_w - 1] = raw_w - 1
        result[:, 4][result[:, 4] > raw_h - 1] = raw_h - 1

        return result


    def complex_call(self,img,iou_thres=0.5,score_thres=0.05):


        result1 =self.four_rotate_call(img,input_size=512)
        result1 = self.py_nms(result1, iou_thres=iou_thres, score_thres=score_thres, max_boxes=1000)
        bbox_1=(result1[:,:4]/1024).tolist()
        score_1=result1[:,4].tolist()
        label_1=result1[:,5].tolist()

        bb,ss,ll=weighted_boxes_fusion([bbox_1],[score_1],[label_1])

        bb=bb*1024
        ss=np.expand_dims(ss,axis=1)
        ll = np.expand_dims(ll, axis=1)

        result=np.concatenate([bb,ss,ll],axis=1)

        return result




    def four_rotate_call(self,image,input_size=640):

        raw_h,raw_w,_=image.shape

        max_edge=max(raw_h,raw_w)

        scale=input_size/max_edge

        img_resized = cv2.resize(image,None, fx=scale,fy=scale,interpolation=cv2.INTER_LINEAR)

        img=np.full(shape=[input_size,input_size,3],fill_value=cfg.DATA.IMAGENET_DEFAULT_MEAN)



        resized_h,resized_w,_=img_resized.shape
        img[:resized_h,:resized_w,:]=img_resized

        img_rotate_90=np.rot90(img,1)
        img_rotate_180 = np.rot90(img, 2)
        img_rotate_270 = np.rot90(img, 3)

        img_flip=np.fliplr(img)
        img_flip_rotate_90 = np.rot90(img_flip,1)
        img_flip_rotate_180 = np.rot90(img_flip,2)
        img_flip_rotate_270 = np.rot90(img_flip,3)


        image_input=np.stack([img,img_rotate_90,img_rotate_180,img_rotate_270,\
                              img_flip,img_flip_rotate_90,img_flip_rotate_180,img_flip_rotate_270])

        image_input=np.transpose(image_input,axes=[0,3,1,2])

        data = torch.from_numpy(image_input).to(self.device).float()


        data = data.float().sub_(self.mean).div_(self.std)

        with torch.no_grad():
            output=self.model(data,input_size)

        output=output.cpu().numpy()

        output[1] = self.Rotate_with_box(img,angle=-90,boxes=output[1])

        output[2] = self.Rotate_with_box(img, angle=-180, boxes=output[2])

        output[3] = self.Rotate_with_box(img, angle=-270, boxes=output[3])

        # # ###flip
        #
        output[4]=self.Flip_with_box(img,output[4])

        output[5] = self.Rotate_with_box(img, angle=-90, boxes=output[5])
        output[5] = self.Flip_with_box(img, output[5])

        output[6] = self.Rotate_with_box(img, angle=-180, boxes=output[6])
        output[6] = self.Flip_with_box(img, output[6])

        output[7] = self.Rotate_with_box(img, angle=-270, boxes=output[7])
        output[7] = self.Flip_with_box(img, output[7])

        ###cat
        result=output.reshape([-1,6])

        ###scale back to raw image
        result[:, 0:4] = result[:, 0:4] / scale

        ##clip
        result[:, 0:2][result[:, 0:2]<0]=0

        result[:, 3][result[:,3] >raw_w-1] = raw_w-1
        result[:, 4][result[:, 4] > raw_h - 1] = raw_h - 1


        return result


    def Flip_with_box(self,src,boxes):

        h,w,_=src.shape
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 0] = xmin
        boxes[:, 2] = xmax
        return  boxes

    def Rotate_with_box(self,src, angle, boxes=None, center=None, scale=1.0):
        '''
        :param src: src image
        :param label: label should be numpy array with [[x1,y1],
                                                        [x2,y2],
                                                        [x3,y3]...]
        :param angle:angel
        :param center:
        :param scale:
        :return: the rotated image and the points
        '''

        def Rotate_coordinate(label, rt_matrix):
            if rt_matrix.shape[0] == 2:
                rt_matrix = np.row_stack((rt_matrix, np.asarray([0, 0, 1])))
            full_label = np.row_stack((label, np.ones(shape=(1, label.shape[1]))))
            label_rotated = np.dot(rt_matrix, full_label)
            label_rotated = label_rotated[0:2, :]
            return label_rotated

        def box_to_point(boxes):
            '''

            :param boxes: [n,x,y,x,y]
            :return: [4n,x,y]
            '''
            ##caution the boxes are ymin xmin ymax xmax
            points_set = np.zeros(shape=[4 * boxes.shape[0], 2])

            for i in range(boxes.shape[0]):
                points_set[4 * i] = np.array([boxes[i][0], boxes[i][1]])
                points_set[4 * i + 1] = np.array([boxes[i][0], boxes[i][3]])
                points_set[4 * i + 2] = np.array([boxes[i][2], boxes[i][3]])
                points_set[4 * i + 3] = np.array([boxes[i][2], boxes[i][1]])

            return points_set

        def point_to_box(points):
            boxes = []
            points = points.reshape([-1, 4, 2])

            for i in range(points.shape[0]):
                box = [np.min(points[i][:, 0]), np.min(points[i][:, 1]), np.max(points[i][:, 0]),
                       np.max(points[i][:, 1])]

                boxes.append(box)

            return np.array(boxes)

        boxes_raw=np.array(boxes)

        boxes=np.array(boxes[:,0:4])


        label = box_to_point(boxes)
        image = src
        (h, w) = image.shape[:2]
        # 若未指定旋转中心，则将图像中心设为旋转中心

        if center is None:
            center = (w / 2, h / 2)
        # 执行旋转
        M = cv2.getRotationMatrix2D(center, angle, scale)

        new_size = Rotate_coordinate(np.array([[0, w, w, 0],
                                               [0, 0, h, h]]), M)

        new_h, new_w = np.max(new_size[1]) - np.min(new_size[1]), np.max(new_size[0]) - np.min(new_size[0])

        scale = min(h / new_h, w / new_w)

        M = cv2.getRotationMatrix2D(center, angle, scale)

        label = label.T
        ####make it as a 3x3 RT matrix
        full_M = np.row_stack((M, np.asarray([0, 0, 1])))
        ###make the label as 3xN matrix
        full_label = np.row_stack((label, np.ones(shape=(1, label.shape[1]))))
        label_rotated = np.dot(full_M, full_label)
        label_rotated = label_rotated[0:2, :]
        # label_rotated = label_rotated.astype(np.int32)
        label_rotated = label_rotated.T

        boxes_rotated = point_to_box(label_rotated)

        boxes_raw[:,0:4]=boxes_rotated
        return  boxes_raw

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
