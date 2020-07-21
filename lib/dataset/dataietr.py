


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
    def __init__(self,img_root,txt):
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

            _img_path=line.rsplit('| ',1)[0]
            _label=line.rsplit('| ',1)[-1]

            current_img_path=os.path.join(self.root_path,_img_path)
            current_img_label=_label
            self.metas.append([current_img_path,current_img_label])

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

        self.transform=A.Compose(
                                [
                                    A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
                                    A.OneOf([
                                        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
                                                             val_shift_limit=0.2, p=0.9),
                                        A.RandomBrightnessContrast(brightness_limit=0.2,
                                                                   contrast_limit=0.2, p=0.9),
                                    ],p=0.9),
                                    A.ToGray(p=0.01),
                                    A.HorizontalFlip(p=0.5),
                                    A.VerticalFlip(p=0.5),
                                    A.Resize(height=cfg.DATA.hin, width=cfg.DATA.win, p=1),
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

        ann_info = data_info(im_root_path, ann_file)
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


        if sample_dice > 0.7 and sample_dice <= 1:
            image, boxes = Random_scale_withbbox(image, boxes, target_shape=[cfg.DATA.hin, cfg.DATA.win],
                                                 jitter=0.3)
        elif sample_dice > 0.35 and sample_dice <= 0.7:
            boxes_ = boxes[:, 0:4]
            klass_ = boxes[:, 4:]

            image, boxes_, klass_ = dsfd_aug(image, boxes_, klass_)

            image = image.astype(np.uint8)
            boxes = np.concatenate([boxes_, klass_], axis=1)
        else:
            boxes_ = boxes[:, 0:4]
            klass_ = boxes[:, 4:]
            image, boxes_, klass_ = baidu_aug(image, boxes_, klass_)

            image = image.astype(np.uint8)
            boxes = np.concatenate([boxes_, klass_], axis=1)


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

    def _map_func(self,dp,is_training):
        """Data augmentation function."""
        ####customed here
        try:

            if is_training:
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





    def random_affine(self,img, targets=(), degrees=0, translate=.0, scale=.5, shear=0):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
        # targets = [cls, xyxy]

        height = img.shape[0]   # shape(h,w,c)
        width = img.shape[1]

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(-translate, translate) * img.shape[1]   # x translation (pixels)
        T[1, 2] = random.uniform(-translate, translate) * img.shape[0]   # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Combined rotation matrix
        M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
        if  (M != np.eye(3)).any():  # image changed
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=cfg.DATA.IMAGENET_DEFAULT_MEAN)

        # Transform label coordinates
        n = len(targets)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # # apply angle-based reduction of bounding boxes
            # radians = a * math.pi / 180
            # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            # x = (xy[:, 2] + xy[:, 0]) / 2
            # y = (xy[:, 3] + xy[:, 1]) / 2
            # w = (xy[:, 2] - xy[:, 0]) * reduction
            # h = (xy[:, 3] - xy[:, 1]) * reduction
            # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            area0 = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
            i = (w > 2) & (h > 2) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 20)

            targets = targets[i]
            targets[:, 0:4] = xy[i]

        return img, targets


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

