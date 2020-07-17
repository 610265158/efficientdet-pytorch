
from lib.core.base_trainer.net_work import Train
from lib.dataset.dataietr import DataIter

import cv2
import numpy as np

from train_config import config as cfg
import setproctitle


setproctitle.setproctitle(cfg.MODEL.model_name)


def main():



    ###build dataset
    train_ds = DataIter(cfg.DATA.root_path, cfg.DATA.train_txt_path, True)
    test_ds = DataIter(cfg.DATA.root_path, cfg.DATA.val_txt_path, False)


    ###build trainer
    trainer = Train(train_ds=train_ds,val_ds=test_ds)

    print('it is here')
    if cfg.TRAIN.vis:
        print('show it, here')
        for step in range(train_ds.size):

            images, boxes,labels=train_ds()
            print('xxx')
            print(images.shape)

            for i in range(images.shape[0]):
                example_image=np.array(images[i],dtype=np.uint8)
                example_image=np.transpose(example_image,[1,2,0])
                example_boxes=np.array(boxes[i])

                print(example_boxes)

                for k in range(example_boxes.shape[0]):
                    _box=example_boxes[k]

                    xmin = int(_box[0])
                    ymin = int(_box[1])
                    xmax = int(_box[2])
                    ymax = int(_box[3])

                    example_image=np.ascontiguousarray(example_image)
                    print(example_image.shape)
                    cv2.rectangle(example_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
                cv2.imshow('example',example_image)


                cv2.waitKey(0)

    ### train
    trainer.custom_loop()

if __name__=='__main__':
    main()