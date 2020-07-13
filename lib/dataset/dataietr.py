


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

            ### do crazy crop

            # if random.uniform(0,1)<cfg.DATA.cracy_crop and self.traing_flag:
            #
            #     if len(holder) == self.batch_size:
            #         crazy_holder=[]
            #         for i in range(0,len(holder),4):
            #
            #
            #
            #             ### do random crop 4 times:
            #             for j in range(4):
            #
            #                 curboxes=tmp_bbox.copy()
            #                 cur_klasses=tmp_klass.copy()
            #                 start_h=random.randint(0,cfg.DATA.hin)
            #                 start_w = random.randint(0, cfg.DATA.win)
            #
            #                 cur_img_block=np.array(crazy_iamge[start_h:start_h+cfg.DATA.hin,start_w:start_w+cfg.DATA.win,:])
            #
            #                 for k in range(len(curboxes)):
            #                     curboxes[k][0] = curboxes[k][0] - start_w
            #                     curboxes[k][1] = curboxes[k][1] - start_h
            #                     curboxes[k][2] = curboxes[k][2] - start_w
            #                     curboxes[k][3] = curboxes[k][3] - start_h
            #
            #                 curboxes[:,[0, 2]] = np.clip(curboxes[:,[0, 2]], 0, cfg.DATA.win - 1)
            #                 curboxes[:,[1, 3]] = np.clip(curboxes[:,[1, 3]], 0, cfg.DATA.hin - 1)
            #                 ###cove the small faces
            #
            #
            #                 boxes_clean=[]
            #                 klsses_clean=[]
            #                 for k in range(curboxes.shape[0]):
            #                     box = curboxes[k]
            #
            #                     if not ((box[3] - box[1]) < cfg.DATA.cover_obj or (
            #                             box[2] - box[0]) < cfg.DATA.cover_obj):
            #
            #                         boxes_clean.append(curboxes[k])
            #                         klsses_clean.append(cur_klasses[k])
            #
            #                 boxes_clean=np.array(boxes_clean)
            #                 klsses_clean=np.array(klsses_clean)
            #
            #
            #                 crazy_holder.append([cur_img_block,boxes_clean,klsses_clean])
            #
            #
            #
            #         holder=crazy_holder


            if len(holder) == self.batch_size:

                padd_holder = []
                for j,item in enumerate(holder):
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

        self.space_augmentor = Sequence([RandomShear()])
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

    def _map_func(self,dp,is_training):
        """Data augmentation function."""
        ####customed here
        try:
            fname, annos = dp
            image = cv2.imread(fname, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            labels = annos.split(' ')
            boxes = []


            for label in labels:
                bbox = np.array(label.split(','), dtype=np.float)
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]])

            boxes = np.array(boxes, dtype=np.float)


            if is_training:

                sample_dice = random.uniform(0, 1)
                if sample_dice > 0.75 and sample_dice <= 1:
                    image, boxes = Random_scale_withbbox(image, boxes, target_shape=[cfg.DATA.hin, cfg.DATA.win],
                                                         jitter=0.3)
                elif sample_dice > 0.5 and sample_dice <= 0.75:
                    boxes_ = boxes[:, 0:4]
                    klass_ = boxes[:, 4:]

                    image, boxes_, klass_ = dsfd_aug(image, boxes_, klass_)

                    image = image.astype(np.uint8)
                    boxes = np.concatenate([boxes_, klass_], axis=1)
                elif sample_dice > 0.25 and sample_dice <= 0.5:
                    boxes_ = boxes[:, 0:4]
                    klass_ = boxes[:, 4:]
                    image, boxes_, klass_ = baidu_aug(image, boxes_, klass_)

                    image = image.astype(np.uint8)
                    boxes = np.concatenate([boxes_, klass_], axis=1)
                else:
                    pass

                if random.uniform(0, 1) > 0.5:
                    image, boxes = Random_flip(image, boxes)

                if random.uniform(0, 1) > 0.5:
                    boxes_ = boxes[:, 0:4]
                    klass_ = boxes[:, 4:]
                    angel=random.choice([90,180,270])

                    image, boxes_ = Rotate_with_box(image,angel, boxes_)
                    boxes = np.concatenate([boxes_, klass_], axis=1)




                if random.uniform(0, 1) > 0.5:
                    image =self.color_augmentor(image)

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
                image = cv2.resize(image, (cfg.DATA.win, cfg.DATA.hin))

                boxes_[:, 0] *= cfg.DATA.win
                boxes_[:, 1] *= cfg.DATA.hin
                boxes_[:, 2] *= cfg.DATA.win
                boxes_[:, 3] *= cfg.DATA.hin
                image = image.astype(np.uint8)
                boxes = np.concatenate([boxes_, klass_], axis=1)

                if random.uniform(0, 1) < cfg.DATA.cracy_crop:
                    image,boxes=self.crazy_crop(image, boxes)

                if random.uniform(0, 1) > 0.5:
                    image =pixel_jitter(image,15)
            else:
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
                image = cv2.resize(image, (cfg.DATA.win, cfg.DATA.hin))

                boxes_[:, 0] *= cfg.DATA.win
                boxes_[:, 1] *= cfg.DATA.hin
                boxes_[:, 2] *= cfg.DATA.win
                boxes_[:, 3] *= cfg.DATA.hin
                image = image.astype(np.uint8)
                boxes = np.concatenate([boxes_, klass_], axis=1)




            if boxes.shape[0] == 0 or np.sum(image) == 0:
                boxes_ = np.array([[0, 0, 0, 0]])
                klass_ = np.array([0])
            else:
                boxes_ = np.array(boxes[:, 0:4], dtype=np.float32)
                klass_ = np.array(boxes[:, 4], dtype=np.int64)


        except:
            logger.warn('there is an err with %s' % fname)
            traceback.print_exc()
            image = np.zeros(shape=(cfg.DATA.hin, cfg.DATA.win, 3), dtype=np.float32)
            boxes_ = np.array([[0, 0, 0, 0]])
            klass_ = np.array([0])


        return image, boxes_, klass_

    def crazy_crop(self,image,boxes):
        holder=[[image,boxes[:,0:4],boxes[:,4:5]]]


        three_item=random.sample(self.lst,3)

        for i in range(len(three_item)):
            fname, annos = three_item[i]

            image = cv2.imread(fname, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            labels = annos.split(' ')
            boxes = []

            for label in labels:
                bbox = np.array(label.split(','), dtype=np.float)
                boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]])

            boxes = np.array(boxes, dtype=np.float)

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
            image = cv2.resize(image, (cfg.DATA.win, cfg.DATA.hin))

            boxes_[:, 0] *= cfg.DATA.win
            boxes_[:, 1] *= cfg.DATA.hin
            boxes_[:, 2] *= cfg.DATA.win
            boxes_[:, 3] *= cfg.DATA.hin
            image = image.astype(np.uint8)
            boxes = np.concatenate([boxes_, klass_], axis=1)

            if random.uniform(0, 1) > 0.5:
                image = self.color_augmentor(image)


            holder.append([image,boxes[:,0:4],boxes[:,4:5]])



        crazy_iamge = np.zeros(shape=(2 * cfg.DATA.hin, 2 * cfg.DATA.win, 3), dtype=holder[i][0].dtype)

        crazy_iamge[:cfg.DATA.hin, :cfg.DATA.win, :] = holder[0][0]
        crazy_iamge[:cfg.DATA.hin, cfg.DATA.win:, :] = holder[1][0]
        crazy_iamge[cfg.DATA.hin:, :cfg.DATA.win, :] = holder[2][0]
        crazy_iamge[cfg.DATA.hin:, cfg.DATA.win:, :] = holder[3][0]

        holder[ 1][1][:, [0, 2]] = holder[ 1][1][:, [0, 2]] + cfg.DATA.win

        holder[ 2][1][:, [1, 3]] = holder[ 2][1][:, [1, 3]] + cfg.DATA.hin

        holder[ 3][1][:, [0, 2]] = holder[ 3][1][:, [0, 2]] + cfg.DATA.win
        holder[ 3][1][:, [1, 3]] = holder[ 3][1][:, [1, 3]] + cfg.DATA.hin

        tmp_bbox = np.concatenate((holder[0][1],
                                   holder[1][1],
                                   holder[2][1],
                                   holder[3][1]),
                                  axis=0)

        tmp_klass = np.concatenate((holder[0][2],
                                    holder[1][2],
                                    holder[2][2],
                                    holder[3][2]),
                                   axis=0)
        curboxes = tmp_bbox.copy()
        cur_klasses = tmp_klass.copy()
        start_h = random.randint(0, cfg.DATA.hin)
        start_w = random.randint(0, cfg.DATA.win)

        cur_img_block = np.array(crazy_iamge[start_h:start_h + cfg.DATA.hin, start_w:start_w + cfg.DATA.win, :])

        for k in range(len(curboxes)):
            curboxes[k][0] = curboxes[k][0] - start_w
            curboxes[k][1] = curboxes[k][1] - start_h
            curboxes[k][2] = curboxes[k][2] - start_w
            curboxes[k][3] = curboxes[k][3] - start_h

        curboxes[:, [0, 2]] = np.clip(curboxes[:, [0, 2]], 0, cfg.DATA.win - 1)
        curboxes[:, [1, 3]] = np.clip(curboxes[:, [1, 3]], 0, cfg.DATA.hin - 1)
        ###cove the small faces

        boxes_clean = []
        klsses_clean = []
        for k in range(curboxes.shape[0]):
            box = curboxes[k]

            if not ((box[3] - box[1]) < cfg.DATA.cover_obj or (
                    box[2] - box[0]) < cfg.DATA.cover_obj):
                boxes_clean.append(curboxes[k])
                klsses_clean.append(cur_klasses[k])

        boxes_clean = np.array(boxes_clean)
        klsses_clean = np.array(klsses_clean)


        label=np.concatenate([boxes_clean,klsses_clean],1)

        return cur_img_block, label

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i



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
                                  scale_range=cfg.DATA.scales,
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

