#-*-coding:utf-8-*-



import os
import numpy as np
import random

from easydict import EasyDict as edict





os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = edict()





config.TRAIN = edict()

#### below are params for dataiter
config.TRAIN.process_num = 4
config.TRAIN.prefetch_size = 15



config.TRAIN.num_gpu = 1
config.TRAIN.batch_size = 16
config.TRAIN.log_interval = 10                  ##10 iters for a log msg
config.TRAIN.test_interval = 1
config.TRAIN.epoch = 100

config.TRAIN.init_lr=5.e-4

config.TRAIN.weight_decay_factor = 5.e-4                                   ####l2
config.TRAIN.vis=False
config.TRAIN.mix_precision=True


config.TRAIN.warmup_step=1500
config.TRAIN.opt='Adamw'
config.TRAIN.ema=False


config.DATA = edict()
config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'
config.DATA.mutiscale=False
config.DATA.scale_choice=[(512,16),(640,12),(768,6),(896,4),(1024,4)]
config.DATA.hin = 512                                        # input size during training , 128,160,   depends on
config.DATA.win = 512

config.DATA.IMAGENET_DEFAULT_MEAN = [ x *255 for x in (0.485, 0.456, 0.406)]
config.DATA.IMAGENET_DEFAULT_STD = [x *255 for x in  (0.229, 0.224, 0.225)]
config.DATA.max_boxes=300
config.DATA.cover_obj=5


config.DATA.mixup=0.5
config.MODEL = edict()
config.MODEL.model_name='tf_efficientdet_d2'
config.MODEL.model_path = './models/'                                        ## save directory

config.MODEL.pretrained_model=None
config.MODEL.freeze_bn=False

config.SEED=42


from lib.utils.seed_things import seed_everything

seed_everything(config.SEED)