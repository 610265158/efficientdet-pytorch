from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2

annFile='./Val_cocoStyle.json'
coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))


# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['wheat']);
imgIds = coco.getImgIds(catIds=catIds );

print(imgIds)
show_flag=False
for id in (imgIds):
    img = coco.loadImgs(id)[0]

    print(img)
    # load and display image
    # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
    # use url to load image

    print()
    I = cv2.imread(img['file_name'])


    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)


    for ann in anns:
        box=ann['bbox']

        xmin=int(box[0])
        ymin = int(box[1])
        xmax = int(box[0]+box[2])
        ymax =int(box[1]+box[3])


        if box[2]>200 or box[3]>200:
            show_flag=True
        cv2.rectangle(I,(xmin,ymin),(xmax,ymax),(0,255,0),3)
    if show_flag:
        cv2.namedWindow('ss',0)
        cv2.imshow('ss',I)
        cv2.waitKey(0)
    show_flag=False