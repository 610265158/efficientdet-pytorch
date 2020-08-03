#-*-coding:utf-8-*-

import sklearn.metrics
import cv2
import time
import os
import torch

import torch.nn as nn
import numpy as np
import random
import timm



from train_config import config as cfg


if cfg.TRAIN.mix_precision:
    from apex import amp

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet


from lib.utils.logger import logger

from lib.core.base_trainer.metric import AverageMeter
from lib.utils.torch_utils import EMA


def mixup(data, target_boxes,target_labels, alpha):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_target_boxes = target_boxes[indices]
    shuffled_target_labels = target_labels[indices]

    lam = 0.5
    data = data * lam + shuffled_data * (1 - lam)
    new_target_boxes=torch.cat([target_boxes,shuffled_target_boxes],1)
    new_target_labels = torch.cat([target_labels, shuffled_target_labels], 1)
    return data, new_target_boxes,new_target_labels,lam



class Train(object):
  """Train class.
  """

  def __init__(self,train_ds,val_ds):

    self.init_lr=cfg.TRAIN.init_lr
    self.warup_step=cfg.TRAIN.warmup_step
    self.epochs = cfg.TRAIN.epoch
    self.batch_size = cfg.TRAIN.batch_size
    self.l2_regularization=cfg.TRAIN.weight_decay_factor

    self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    config = get_efficientdet_config(cfg.MODEL.model_name)
    config.num_classes=1
    config.image_size = cfg.DATA.hin

    net = EfficientDet(config, pretrained_backbone=False)


    checkpoint = torch.load(cfg.MODEL.model_name+'.pth')
    #
    # for k, v in checkpoint.items():
    #     if 'predict' in k:
    #         print(k)
    checkpoint.pop('class_net.predict.conv_dw.weight')
    checkpoint.pop('class_net.predict.conv_pw.weight')
    checkpoint.pop('class_net.predict.conv_pw.bias')



    net.load_state_dict(checkpoint,strict=False)

    if cfg.MODEL.pretrained_model is not None:
        state_dict = torch.load(cfg.MODEL.pretrained_model, map_location=self.device)
        net.load_state_dict(state_dict, strict=False)

        
    self.model = DetBenchTrain(net, config)

    self.detector=self.model.to(self.device)


    self.mean = torch.tensor( cfg.DATA.IMAGENET_DEFAULT_MEAN).to(self.device).view(1, 3, 1, 1)
    self.std = torch.tensor(cfg.DATA.IMAGENET_DEFAULT_STD).to(self.device).view(1, 3, 1, 1)

    param_optimizer = list(self.model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': cfg.TRAIN.weight_decay_factor},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if 'Adamw' in cfg.TRAIN.opt:

      self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                         lr=self.init_lr,eps=1.e-5)
    else:
      self.optimizer = torch.optim.SGD(self.model.parameters(),
                                       lr=self.init_lr,
                                       momentum=0.9)



    if cfg.TRAIN.mix_precision:
        self.model, self.optimizer = amp.initialize( self.model, self.optimizer, opt_level="O1")

    if cfg.TRAIN.num_gpu>1:
        self.model=nn.DataParallel(self.model)

    self.ema = EMA(self.model, 0.999)

    self.ema.register()
    ###control vars
    self.iter_num=0

    self.train_ds=train_ds

    self.val_ds = val_ds

    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min',factor=0.5, patience=4,verbose=True,min_lr=1.e-6)
    # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( self.optimizer, self.epochs,eta_min=1.e-6)

    self.accumulation_steps=cfg.TRAIN.accumulation_steps

  def custom_loop(self):
    """Custom training and testing loop.
    Args:
      train_dist_dataset: Training dataset created using strategy.
      test_dist_dataset: Testing dataset created using strategy.
      strategy: Distribution strategy.
    Returns:
      train_loss, train_accuracy, test_loss, test_accuracy
    """

    def distributed_train_epoch(epoch_num):

      summary_loss = AverageMeter()
      self.model.train()
      if cfg.MODEL.freeze_bn:
          for m in self.model.modules():
              if isinstance(m, nn.BatchNorm2d):
                  m.eval()
                  if cfg.MODEL.freeze_bn_affine:
                      m.weight.requires_grad = False
                      m.bias.requires_grad = False
      for step in range(self.train_ds.size):

        if epoch_num<10:
            ###excute warm up in the first epoch
            if self.warup_step>0:
                if self.iter_num < self.warup_step:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.iter_num / float(self.warup_step) * self.init_lr
                        lr = param_group['lr']

                    logger.info('warm up with learning rate: [%f]' % (lr))

        start=time.time()


        if random.uniform(0,1)<cfg.DATA.mixup:
            images, boxes_target, label_target = self.train_ds()

            data = torch.from_numpy(images).to(self.device).float().sub_(self.mean).div_(self.std)
            boxes_target = torch.from_numpy(boxes_target).to(self.device).float()
            label_target = torch.from_numpy(label_target).to(self.device).float()


            alpha=0.5
            data,boxes_target,label_target,alpha=mixup(data,boxes_target,label_target,alpha)

            target = {}
            target['bbox'] = boxes_target
            target['cls'] = label_target
            batch_size = data.shape[0]
            ##xyxy to yxyx
            target['bbox'][:, :, [0, 1, 2, 3]] = target['bbox'][:, :, [1, 0, 3, 2]]

            loss_dict = self.model(data, target)

            current_loss = loss_dict['loss']*alpha
            summary_loss.update(current_loss.detach().item(), batch_size)


        else:

            images, boxes_target,label_target = self.train_ds()


            data = torch.from_numpy(images).to(self.device).float().sub_(self.mean).div_(self.std)
            boxes_target = torch.from_numpy(boxes_target).to(self.device).float()
            label_target= torch.from_numpy(label_target).to(self.device).float()

            target={}
            target['bbox']=boxes_target
            target['cls']=label_target
            batch_size = data.shape[0]
            ##xyxy to yxyx
            target['bbox'][:,:, [0, 1, 2, 3]] = target['bbox'][:,:, [1, 0, 3, 2]]




            if cfg.DATA.mutiscale:
                cur_shape=data.shape[3]

                logger.info('training with shape %dx%d'%(cur_shape,cur_shape))

            loss_dict = self.model(data,target)

            current_loss=loss_dict['loss']
            summary_loss.update(current_loss.detach().item(), batch_size)

        current_loss=current_loss/self.accumulation_steps



        if cfg.TRAIN.mix_precision:
            with amp.scale_loss(current_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            current_loss.backward()

        if ((step + 1) % self.accumulation_steps) == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()


        if cfg.TRAIN.ema:
            self.ema.update()
        self.iter_num+=1
        time_cost_per_batch=time.time()-start

        images_per_sec=cfg.TRAIN.batch_size/time_cost_per_batch


        if self.iter_num%cfg.TRAIN.log_interval==0:

            log_message = 'Train Step %d, ' \
                          'summary_loss: %.6f, ' \
                          'time: %.6f, '\
                          'speed %d images/persec'% (
                              step, summary_loss.avg, time.time() - start,images_per_sec)
            logger.info(log_message)




      return summary_loss

    def distributed_test_epoch(epoch_num):
        summary_loss = AverageMeter()



        self.model.eval()
        t = time.time()
        with torch.no_grad():
            for step in range(self.val_ds.size):
                images, boxes_target, label_target = self.val_ds()

                data = torch.from_numpy(images).to(self.device).float().sub_(self.mean).div_(self.std)
                boxes_target = torch.from_numpy(boxes_target).to(self.device).float()
                label_target = torch.from_numpy(label_target).to(self.device).float()
                target = {}

                target['bbox'] = boxes_target


                target['cls'] = label_target
                batch_size = data.shape[0]

                target['bbox'][:,:, [0, 1, 2, 3]] = target['bbox'][:,:, [1, 0, 3, 2]]

                loss_dict = self.model(data, target)
                current_loss = loss_dict['loss']

                summary_loss.update(current_loss.detach().item(), batch_size)

                if step % cfg.TRAIN.log_interval == 0:

                    log_message = 'Val Step %d, ' \
                                  'summary_loss: %.6f, ' \
                                  'time: %.6f' % (
                                  step, summary_loss.avg, time.time() - t)

                    logger.info(log_message)


        return summary_loss

    for epoch in range(self.epochs):

      for param_group in self.optimizer.param_groups:
        lr=param_group['lr']
      logger.info('learning rate: [%f]' %(lr))
      t=time.time()

      summary_loss= distributed_train_epoch(epoch)

      train_epoch_log_message = '[RESULT]: Train. Epoch: %d,' \
                                ' summary_loss: %.5f,' \
                                ' time:%.5f' % (
                                epoch, summary_loss.avg, (time.time() - t))
      logger.info(train_epoch_log_message)




      ##switch eam weighta
      if cfg.TRAIN.ema:
        self.ema.apply_shadow()
      if epoch%cfg.TRAIN.test_interval==0:

          summary_loss= distributed_test_epoch(epoch)

          val_epoch_log_message = '[RESULT]: VAL. Epoch: %d,' \
                                  ' summary_loss: %.5f,' \
                                  ' time:%.5f' % (
                                   epoch, summary_loss.avg, (time.time() - t))
          logger.info(val_epoch_log_message)

      # self.scheduler.step()
      self.scheduler.step(summary_loss.avg)
      #### save the model every end of epoch

      current_model_saved_name='./models/epoch_%d_val_loss%.6f.pth'%(epoch,summary_loss.avg)

      logger.info('A model saved to %s' % current_model_saved_name)

      if not os.access(cfg.MODEL.model_path,os.F_OK):
        os.mkdir(cfg.MODEL.model_path)


      torch.save(self.model.model.state_dict(),current_model_saved_name)

      ####switch back
      if cfg.TRAIN.ema:
        self.ema.restore()


      # save_checkpoint({
      #           'state_dict': self.model.state_dict(),
      #           },iters=epoch,tag=current_model_saved_name)


    current_model_saved_name = './models/final.pth'

    logger.info('final model saved to %s' % current_model_saved_name)

    torch.save(self.model.state_dict(), current_model_saved_name)



  def load_weight(self):
      if cfg.MODEL.pretrained_model is not None:
          state_dict=torch.load(cfg.MODEL.pretrained_model, map_location=self.device)
          self.model.load_state_dict(state_dict,strict=False)



