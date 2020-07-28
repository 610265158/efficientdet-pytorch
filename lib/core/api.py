#-*-coding:utf-8-*-
import numpy as np
import torch

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain,DetBenchPredict
from effdet.efficientdet import HeadNet
from torchvision.ops.boxes import batched_nms, remove_small_boxes
from train_config import config as cfg
import cv2


class Detector():
    def __init__(self,model_path):


        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        config = get_efficientdet_config(cfg.MODEL.model_name)

        config.num_classes = 1

        net = EfficientDet(config, pretrained_backbone=False)

        state_dict = torch.load(model_path, map_location=self.device)
        net.load_state_dict(state_dict, strict=False)


        self.model = DetBenchPredict(net, config)

        self.model = self.model.to(self.device)

        self.model.eval()


        self.mean = torch.tensor(cfg.DATA.IMAGENET_DEFAULT_MEAN).to(self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg.DATA.IMAGENET_DEFAULT_STD).to(self.device).view(1, 3, 1, 1)


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


        scale_tensor=torch.tensor([1/scale]).to(self.device)
        image_size_tensor=torch.tensor([[max_edge,max_edge]]).to(self.device)
        with torch.no_grad():
            output=self.model(data,scale_tensor,image_size_tensor)

            output=output[0]

            boxes=output[:,0:4]
            scores=output[:,4]
            labels=output[:,5]

            detections=self.nms(boxes,scores,labels)
        output=detections.cpu().numpy()

        result=output

        return result




    def simple_call(self, image, input_size=640):
        raw_h, raw_w, _ = image.shape

        max_edge = max(raw_h, raw_w)

        scale = input_size / max_edge

        img_resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        img = np.full(shape=[input_size, input_size, 3], fill_value=cfg.DATA.IMAGENET_DEFAULT_MEAN)

        resized_h, resized_w, _ = img_resized.shape
        img[:resized_h, :resized_w, :] = img_resized

        img = np.transpose(img, axes=[2, 0, 1])
        img = np.expand_dims(img, 0)
        data = torch.from_numpy(img).to(self.device).float()

        data = data.float().sub_(self.mean).div_(self.std)

        scale_tensor = torch.tensor([1 / scale]).to(self.device)
        image_size_tensor = torch.tensor([[max_edge, max_edge]]).to(self.device)
        with torch.no_grad():
            output = self.model(data, scale_tensor, image_size_tensor)


        return output


    def complex_call(self,img,iou_thrs):


        result1 =self.four_rotate_call(img,input_size=1024)
        
        result=torch.cat([result1],dim=0)
        result = result.reshape([-1, 6])


        boxes = result[:, 0:4]
        scores = result[:, 4]
        labels = result[:, 5]

        detections = self.nms(boxes, scores, labels,iou_thrs)

        output = detections.cpu().numpy()


        return output




    def four_rotate_call(self,image,input_size):

        h,w,c=image.shape
        res1=self.simple_call(image,input_size=input_size)

        fliplr_image=np.fliplr(image)

        res2 = self.simple_call(fliplr_image, input_size=input_size)

        xmin = w - res2[:, :, 2]
        xmax = w - res2[:, :, 0]
        res2[:, :, 0]=xmin
        res2[:, :, 2]=xmax

        flipup_image = np.flipud(image)

        res3 = self.simple_call(flipup_image, input_size=input_size)
        ymin = h - res3[:, :, 3]
        ymax = h - res3[:, :, 1]
        res3[:, :, 1]=ymin
        res3[:, :, 3]=ymax
        result=torch.cat([res1,res2,res3],0)



        return result


    def Flip_with_box(self,src,boxes,scale,up_down=False):



        h,w,_=src.shape


        h=h/scale
        w=w/scale

        if up_down:
            ymin = h - boxes[:, 3]
            ymax = h - boxes[:, 1]
            boxes[:, 1] = ymin
            boxes[:, 3] = ymax

        else:

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

    def nms(self,boxes,scores,classes,iou_thrs=0.5):

        ##we do in the outside
        top_detection_idx = batched_nms(boxes, scores, classes, iou_threshold=iou_thrs)

        boxes=boxes[top_detection_idx,...]
        scores=scores[top_detection_idx,...]
        classes = classes[top_detection_idx, ...]

        scores = scores.unsqueeze(dim=-1)
        classes = classes.unsqueeze(dim=-1)
        detections = torch.cat([boxes, scores, classes.float()], dim=1)
        return detections

    def load_weight(self,path):

        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)
