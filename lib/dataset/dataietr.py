


import os
import random
import cv2
import numpy as np
import traceback

from lib.utils.logger import logger
from tensorpack.dataflow import DataFromGenerator
from tensorpack.dataflow import BatchData, MultiProcessPrefetchData


from lib.dataset.augmentor.augmentation import Random_scale_withbbox,\
                                                Random_flip,\
                                                baidu_aug,\
                                                dsfd_aug,\
                                                Fill_img,\
                                                Rotate_with_box,\
                                                produce_heatmaps_with_bbox,\
                                                box_in_img
from lib.dataset.augmentor.data_aug.bbox_util import *
from lib.dataset.augmentor.data_aug.data_aug import *
from lib.dataset.augmentor.visual_augmentation import ColorDistort,pixel_jitter

from train_config import config as cfg


import math

import albumentations as A
class data_info():
    def __init__(self,img_root,txt,training_flag=False):

        self.training_flag = training_flag


        self.txt_file=txt
        self.root_path = img_root
        self.metas=[]


        self.read_txt()



    def read_txt(self):
        with open(self.txt_file) as _f:
            txt_lines=_f.readlines()
        txt_lines.sort()
        for line in txt_lines:
            line=line.rstrip()


            split_info=line.split('| ')


            source=split_info[0]
            _img_path=split_info[1]
            _label=split_info[2]

            current_img_path=os.path.join(self.root_path,_img_path)
            current_img_label=_label

            if self.training_flag:
                if str(source)=='usask_1' or str(source)=='inrae_1' or str(source)=='arvalis_2':
                    for _ in range(4):
                        self.metas.append([current_img_path, current_img_label])
                else:

                    self.metas.append([current_img_path,current_img_label])
            else:
                self.metas.append([current_img_path, current_img_label])
            ###some change can be made here
        logger.info('the dataset contains %d images'%(len(txt_lines)))
        logger.info('the datasets contains %d samples'%(len(self.metas)))


    def get_all_sample(self):
        random.shuffle(self.metas)
        return self.metas

class MutiScaleBatcher(BatchData):

    def __init__(self, ds, batch_size,
                 remainder=False,
                 use_list=False,
                 scale_range=None,
                 input_size=(512,512),
                 divide_size=32,
                 is_training=True):
        """
        Args:
            ds (DataFlow): A dataflow that produces either list or dict.
                When ``use_list=False``, the components of ``ds``
                must be either scalars or :class:`np.ndarray`, and have to be consistent in shapes.
            batch_size(int): batch size
            remainder (bool): When the remaining datapoints in ``ds`` is not
                enough to form a batch, whether or not to also produce the remaining
                data as a smaller batch.
                If set to False, all produced datapoints are guaranteed to have the same batch size.
                If set to True, `len(ds)` must be accurate.
            use_list (bool): if True, each component will contain a list
                of datapoints instead of an numpy array of an extra dimension.
        """
        super(BatchData, self).__init__(ds)
        if not remainder:
            try:
                assert batch_size <= len(ds)
            except NotImplementedError:
                pass

        self.batch_size = int(batch_size)
        self.remainder = remainder
        self.use_list = use_list

        self.scale_range=scale_range
        self.divide_size=divide_size

        self.input_size=input_size
        self.traing_flag=is_training



    def __iter__(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """

        ##### pick a scale and shape aligment

        holder = []
        for data in self.ds:

            image,boxes_,klass_=data[0],data[1],data[2]

            data=[image,boxes_,klass_]
            holder.append(data)


            if len(holder) == self.batch_size:


                if self.scale_range is not None:
                    cur_shape,cur_batch_size=random.choice(self.scale_range)

                    holder=random.sample(holder,cur_batch_size)

                padd_holder = []
                for j,item in enumerate(holder):
                    if self.scale_range is not None:
                        image=item[0]
                        boxes=item[1]
                        labels=item[2]

                        image, boxes = self.align_resize(image,boxes,target_height=cur_shape,target_width=cur_shape)

                        item[0],item[1]= image, boxes
                    image=np.ascontiguousarray(item[0])

                    image=np.transpose(image,axes=[2,0,1]).astype(np.uint8)
                    box=np.zeros(shape=[cfg.DATA.max_boxes,4])

                    labels = np.zeros(shape=[cfg.DATA.max_boxes])

                    num_objs=len(item[1])
                    if num_objs>0:
                        box[:num_objs]=np.array(item[1])
                        labels[:num_objs]=np.array(item[2])
                    padd_holder.append([image,box,labels])


                yield BatchData.aggregate_batch(padd_holder, self.use_list)


                del padd_holder[:]

                holder=[]

    def place_image(self,img_raw,target_height,target_width):

        channel = img_raw.shape[2]
        raw_height = img_raw.shape[0]
        raw_width = img_raw.shape[1]

        start_h=random.randint(0,target_height-raw_height)
        start_w=random.randint(0,target_width-raw_width)

        img_fill = np.zeros([target_height,target_width,channel], dtype=img_raw.dtype)
        img_fill[start_h:start_h+raw_height,start_w:start_w+raw_width]=img_raw

        return img_fill,start_w,start_h

    def align_resize(self,img_raw,boxes,target_height,target_width):
        ###sometimes use in objs detects
        h, w, c = img_raw.shape


        scale_y = target_height / h
        scale_x = target_width / w

        scale = min(scale_x, scale_y)

        image = cv2.resize(img_raw, None, fx=scale, fy=scale)
        boxes[:,:4]=boxes[:,:4]*scale

        return image, boxes



    def make_safe_box(self,image,boxes):
        h,w,c=image.shape

        boxes[boxes[:,0]<0]=0
        boxes[boxes[:, 1] < 0] = 0
        boxes[boxes[:, 2] >w] = w-1
        boxes[boxes[:, 3] >h] = h-1
        return boxes





class DsfdDataIter():

    def __init__(self, img_root_path='', ann_file=None, training_flag=True, shuffle=True):

        self.color_augmentor = ColorDistort()

        self.training_flag = training_flag

        self.lst = self.parse_file(img_root_path, ann_file)

        self.shuffle = shuffle


        self.no_crop_transform=A.Compose(
                                [
                                    A.OneOf([
                                        A.HueSaturationValue(hue_shift_limit=5,
                                                             sat_shift_limit=5,
                                                             val_shift_limit=5,
                                                             p=0.9),
                                        A.RandomBrightnessContrast(brightness_limit=0.3,
                                                                   contrast_limit=0.3, p=0.9),
                                    ],p=0.9),
                                    A.RGBShift(p=0.9)
                                    # A.ToGray(p=0.01),
                                    #A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
                                ],
                                p=1.0)
        self.transform=A.Compose(
                                [
                                    A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),

                                    # A.ToGray(p=0.01),
                                    A.HorizontalFlip(p=0.5),
                                    A.VerticalFlip(p=0.5),
                                    A.Resize(height=cfg.DATA.hin, width=cfg.DATA.win, p=1),
                                    A.OneOf([
                                        A.HueSaturationValue(hue_shift_limit=5,
                                                             sat_shift_limit=5,
                                                             val_shift_limit=5,
                                                             p=0.9),
                                        A.RandomBrightnessContrast(brightness_limit=0.2,
                                                                   contrast_limit=0.2, p=0.9),
                                    ], p=0.9),
                                    A.RGBShift(p=0.9)
                                    #A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),

                                ],
                                p=1.0,
                                bbox_params=A.BboxParams(
                                    format='pascal_voc',
                                    min_area=0,
                                    min_visibility=0,
                                    label_fields=['labels']
                                ))
    def __iter__(self):
        idxs = np.arange(len(self.lst))

        while True:
            if self.shuffle:
                np.random.shuffle(idxs)
            for k in idxs:
                yield self._map_func(self.lst[k], self.training_flag)

    def __len__(self):
        return len(self.lst)

    def parse_file(self,im_root_path,ann_file):
        '''
        :return: [fname,lbel]     type:list
        '''
        logger.info("[x] Get dataset from {}".format(im_root_path))

        ann_info = data_info(im_root_path, ann_file,self.training_flag)
        all_samples = ann_info.get_all_sample()

        return all_samples



    def _sample(self,dp,is_training):
        pass

    def cv2_read_rgb(self,fname):
        image = cv2.imread(fname, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def align_and_resize(self, image, boxes):
        boxes_ = boxes[:, 0:4]
        klass_ = boxes[:, 4:]
        image, shift_x, shift_y = Fill_img(image, target_width=cfg.DATA.win, target_height=cfg.DATA.hin)
        boxes_[:, 0:4] = boxes_[:, 0:4] + np.array([shift_x, shift_y, shift_x, shift_y], dtype='float32')
        h, w, _ = image.shape
        boxes_[:, 0] /= w
        boxes_[:, 1] /= h
        boxes_[:, 2] /= w
        boxes_[:, 3] /= h
        image = image.astype(np.uint8)
        image = cv2.resize(image, (cfg.DATA.win, cfg.DATA.hin),interpolation=cv2.INTER_LINEAR)

        boxes_[:, 0] *= cfg.DATA.win
        boxes_[:, 1] *= cfg.DATA.hin
        boxes_[:, 2] *= cfg.DATA.win
        boxes_[:, 3] *= cfg.DATA.hin
        image = image.astype(np.uint8)
        boxes = np.concatenate([boxes_, klass_], axis=1)

        return image, boxes

    def eval_sample(self,dp):
        fname, annos = dp
        image = self.cv2_read_rgb(fname)
        labels = annos.split(' ')
        boxes = []

        for label in labels:
            bbox = np.array(label.split(','), dtype=np.float)
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]])

        boxes = np.array(boxes, dtype=np.float)

        image, boxes = self.align_and_resize(image, boxes)
        return image,boxes

    def random_crop_sample(self,dp):
        fname, annos = dp
        image = self.cv2_read_rgb(fname)
        labels = annos.split(' ')
        boxes = []

        for label in labels:
            bbox = np.array(label.split(','), dtype=np.float)
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]])

        boxes = np.array(boxes, dtype=np.float)

        sample_dice = random.uniform(0, 1)

        boxes_ = boxes[:, 0:4]
        klass_ = boxes[:, 4:]

        image, boxes_, klass_ = dsfd_aug(image, boxes_, klass_)

        image = image.astype(np.uint8)
        boxes = np.concatenate([boxes_, klass_], axis=1)


        image,boxes=self.align_and_resize(image,boxes)



        clip_max=image.shape[0]

        boxes[:,0:4]=np.clip(boxes[:,0:4],0,clip_max)
        return image,boxes

    def crazy_crop(self, dp):
        items = [dp]

        items += random.sample(self.lst, 3)

        holder = []
        for i in range(len(items)):
            cur_dp = items[i]
            image, boxes = self.eval_sample(cur_dp)

            holder.append([image, boxes])

        s = cfg.DATA.hin


        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in [-cfg.DATA.hin//2,-cfg.DATA.win//2]]  # mosaic center x, y
        labels4=[]
        for i, item in enumerate(holder):
            # Load image
            img=item[0]
            h,w,_=img.shape

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), cfg.DATA.IMAGENET_DEFAULT_MEAN, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            x = item[1]
            labels = x.copy()
            if x.size > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 0] = x[:, 0] + padw
                labels[:, 1] = x[:, 1] + padh
                labels[:, 2] = x[:, 2] + padw
                labels[:, 3] = x[:, 3] + padh
            labels4.append(labels)

        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            np.clip(labels4[:, 0:4], 0, 2 * s, out=labels4[:, 0:4])  # use with random_affine

        image=img4
        boxes=labels4


        return image,boxes

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


    def load_cutmix_image_and_boxes(self, dp, imsize=1024):
        """
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        dps = [dp] + random.sample(self.lst,3)

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []
        for i, dp in enumerate(dps):
            image, boxes = self.load_image_and_boxes(dp)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)


        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[
            np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]
        return result_image.astype(np.uint8), result_boxes
    def crazy_mix(self,image,boxes,labels):

        def fakeIoU(bbox, gt):
            """
            :param bbox: (n, 4)
            :param gt: (m, 4)
            :return: (n, m)
            numpy 广播机制 从后向前对齐。 维度为1 的可以重复等价为任意维度
            eg: (4,3,2)   (3,2)  (3,2)会扩充为(4,3,2)
                (4,1,2)   (3,2) (4,1,2) 扩充为(4, 3, 2)  (3, 2)扩充为(4, 3,2) 扩充的方法为重复
            广播会在numpy的函数 如sum, maximun等函数中进行
            pytorch同理。
            扩充维度的方法：
            eg: a  a.shape: (3,2)  a[:, None, :] a.shape: (3, 1, 2) None 对应的维度相当于newaxis
            """
            lt = np.maximum(bbox[:, None, :2], gt[:, :2])  # left_top (x, y)
            rb = np.minimum(bbox[:, None, 2:], gt[:, 2:])  # right_bottom (x, y)
            wh = np.maximum(rb - lt + 1, 0)  # inter_area (w, h)
            inter_areas = wh[:, :, 0] * wh[:, :, 1]  # shape: (n, m)
            box_areas = (bbox[:, 2] - bbox[:, 0] + 1) * (bbox[:, 3] - bbox[:, 1] + 1)

            IoU = inter_areas / (box_areas[:, None])
            return IoU

        dp2=random.choice(self.lst)
        if random.uniform(0,1)>0.5:
            image_2,boxes_2=self.load_cutmix_image_and_boxes(dp2)
        else:
            image_2, boxes_2 = self.load_image_and_boxes(dp2)


        boxes_mix=[]
        ###maskout boxes in image1
        # for k in range(boxes.shape[0]):
        #     cur_box=boxes[k].astype(np.int)
        #     image[cur_box[1]:cur_box[3],cur_box[0]:cur_box[2],:]=np.array(cfg.DATA.IMAGENET_DEFAULT_MEAN)

        for k in range(boxes_2.shape[0]):

            cur_box=boxes_2[k].astype(np.int)
            cur_patch=image_2[cur_box[1]:cur_box[3],cur_box[0]:cur_box[2],:]

            cur_width=cur_box[2]-cur_box[0]
            cur_height=cur_box[3]-cur_box[1]


            image_h,image_w,c=image.shape
            start_x=random.randint(0,image_w-cur_width)
            start_y = random.randint(0, image_h - cur_height)

            image[start_y:(start_y+cur_height),start_x:(start_x+cur_width),:]=cur_patch

            box_produced=np.array([start_x,start_y,start_x+cur_width,start_y+cur_height,1])


            boxes_mix.append(box_produced)


        boxes_mix=np.array(boxes_mix)




        ###计算重叠面积， 如果第一个图的重叠面积和自己的面积比超过0.7 可能就需要过滤了

        if boxes.shape[0]==0:
            boxes_produce=boxes_mix
        else:
            try:
                fake_iou=fakeIoU(boxes[:,0:4],boxes_mix[:,0:4])
            except:
                print(boxes.shape)
                print(boxes_mix.shape)
            bool_choosen=np.max(fake_iou,axis=1)<0.7

            box_remain=boxes[bool_choosen]

            klasses_remain=labels[bool_choosen]
            ####
            boxes_1=np.concatenate([box_remain,klasses_remain],axis=1)



            boxes_produce=np.concatenate([boxes_1,boxes_mix])
        #

        return image,boxes_produce[:,:4],boxes_produce[:,4:5]
    def _map_func(self,dp,is_training):
        """Data augmentation function."""
        ####customed here
        try:

            if is_training:

                if random.uniform(0, 1) < cfg.DATA.anchor_based_sample:
                    image, boxes=self.random_crop_sample(dp)

                    transformed=self.no_crop_transform(**{
                        'image': image,
                        'bboxes': boxes[:,0:4],
                        'labels': boxes[:,4]
                    })

                    image = np.array(transformed['image'])
                    boxes_ = np.array(transformed['bboxes'])
                    klasses_ = np.expand_dims(np.array(transformed['labels']), 1)

                    if random.uniform(0, 1)<0.5:
                        image,boxes_=Random_flip(image,boxes_,updown=False)
                    if random.uniform(0, 1)<0.5:
                        image,boxes_=Random_flip(image,boxes_,updown=True)

                else:
                    if random.uniform(0, 1) > 0.5:
                        image, boxes = self.load_cutmix_image_and_boxes(dp)

                    else:
                        image, boxes = self.load_image_and_boxes(dp)

                    transformed=self.transform(**{
                        'image': image,
                        'bboxes': boxes[:,0:4],
                        'labels': boxes[:,4]
                    })


                    image=np.array(transformed['image'])
                    boxes_=np.array(transformed['bboxes'])
                    klasses_=np.expand_dims(np.array(transformed['labels']),1)

                if random.uniform(0, 1) < cfg.DATA.crazy_mix:
                    image, boxes_,klasses_=self.crazy_mix(image,boxes_,klasses_)


                if random.uniform(0, 1) <cfg.DATA.rotate:
                    angle_choice=random.choice([0,90,180,270])
                    image,boxes_=Rotate_with_box(image,angle_choice,boxes_)
                    if cfg.DATA.rotate_jitter>0:
                        angle_choice = random.uniform(0,cfg.DATA.rotate_jitter)
                        image, boxes_ = Rotate_with_box(image, angle_choice, boxes_)

                if random.uniform(0, 1) < cfg.DATA.rgbshuffle:

                    shuffle_ord=[[0,1,2],[2,1,0]]
                    ord_choice=random.choice(shuffle_ord)
                    image[:,:,:]=image[:,:,ord_choice]

                if random.uniform(0, 1) < cfg.DATA.randomquality:

                    quality_choice=random.randint(50,100)

                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality_choice]
                    result, image = cv2.imencode('.jpg', image, encode_param)
                    image = cv2.imdecode(image, 1)

                h_limit,w_limit,_=image.shape

                boxes_=np.clip(boxes_,0,cfg.DATA.hin)

                ###clean the box
                filtered_box=[]
                filtered_klass=[]
                for bb in range(boxes_.shape[0]):
                    cur_box=boxes_[bb,...]
                    cur_kalss=klasses_[bb]
                    bbox_width=cur_box[2]-cur_box[0]
                    bbox_height = cur_box[3] - cur_box[1]


                    if bbox_width*bbox_height<5*5:
                        continue
                    elif bbox_width/bbox_height<0.1 or bbox_width/bbox_height>10 :
                        
                        continue
                    else:
                        filtered_box.append(cur_box)
                        filtered_klass.append(cur_kalss)

                boxes_ = np.array(filtered_box)
                klasses_ = np.array(filtered_klass)

                ##### below process is litlle bit ugly, but it is ok now
                if klasses_.shape[0]==0:
                    boxes_ = np.array([[0, 0, 0, 0]])
                    klasses_ = np.array([[0]])

                boxes = np.concatenate([boxes_, klasses_], axis=1)

                ###### if label is 0 set box as 0,0,0,0
                boxes = boxes * boxes[:, 4:5]

            else:

               image,boxes=self.eval_sample(dp)



            if boxes.shape[0] == 0 or np.sum(image) == 0:
                boxes_ = np.array([[0, 0, 0, 0]])
                klass_ = np.array([0])
            else:
                boxes_ = np.array(boxes[:, 0:4], dtype=np.float32)
                klass_ = np.array(boxes[:, 4], dtype=np.int64)


        except:
            logger.warn('there is an err with %s' % dp[0])
            traceback.print_exc()
            image = np.zeros(shape=(cfg.DATA.hin, cfg.DATA.win, 3), dtype=np.float32)
            boxes_ = np.array([[0, 0, 0, 0]])
            klass_ = np.array([0])


        return image, boxes_, klass_


class DataIter():
    def __init__(self, img_root_path='', ann_file=None, training_flag=True):

        self.shuffle = True
        self.training_flag = training_flag

        self.num_gpu = cfg.TRAIN.num_gpu
        self.batch_size = cfg.TRAIN.batch_size
        self.process_num = cfg.TRAIN.process_num
        self.prefetch_size = cfg.TRAIN.prefetch_size

        self.generator = DsfdDataIter(img_root_path, ann_file, self.training_flag )

        self.ds = self.build_iter()

        self.size=len(self.generator)//self.batch_size

    def parse_file(self, im_root_path, ann_file):

        raise NotImplementedError("you need implemented the parse func for your data")

    def build_iter(self,):


        ds = DataFromGenerator(self.generator)

        if cfg.DATA.mutiscale and self.training_flag:
            ds = MutiScaleBatcher(ds, self.num_gpu * self.batch_size,
                                  scale_range=cfg.DATA.scale_choice,
                                  input_size=(cfg.DATA.hin, cfg.DATA.win),
                                  is_training=self.training_flag)
        else:
            ds = MutiScaleBatcher(ds, self.num_gpu * self.batch_size,
                                  input_size=(cfg.DATA.hin, cfg.DATA.win),
                                  is_training=self.training_flag)
        if not self.training_flag:
            self.process_num=1
        ds = MultiProcessPrefetchData(ds, self.prefetch_size, self.process_num)
        ds.reset_state()
        ds = ds.get_data()
        return ds

    def __call__(self):
        one_batch = next(self.ds)

        return one_batch[0],one_batch[1],one_batch[2]

    def __next__(self):
        return next(self.ds)

    def __len__(self):
        return len(self.generator)

