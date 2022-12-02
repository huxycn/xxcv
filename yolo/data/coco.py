#encoding:utf-8
#
#created by xiongzihua
#
'''
txt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标
'''
import os
import sys
import os.path as osp

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2
import matplotlib.pyplot as plt

from pycocotools.coco import COCO

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# from hutils import cv_utils as cv


# def draw_bboxes_on_img(_img, _bboxes, _labels):
#     for bbox, label in zip(_bboxes, _labels):
#         print(bbox, label)
#         cv.draw_bbox2d(_img, bbox[:2], bbox[2:], cv.RED, label=f'{label}')


# def xywh_to_xyx2y2(xywh):
#     x, y, w, h = xywh
#     return [x, y, x+w, y+h]


def remove_useless_info(coco):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(coco, COCO):
        dataset = coco.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in coco.dataset:
            for anno in coco.dataset["annotations"]:
                anno.pop("segmentation", None)


class COCODetectionDataset(data.Dataset):
    # image_size = 448
    def __init__(self, 
                 img_root, 
                 ann_file, 
                 train, 
                 transform,
                 img_size=(448, 448)):
        print('data init')
        self.img_root = img_root
        self.ann_file = ann_file
        self.train = train
        self.transform=transform
        self.img_size = img_size

        # self.file_names = []
        # self.all_bboxes = []
        # self.all_labels = []
        # self.mean = (123,117,104)#RGB

        # read ann file
        self.coco = COCO(self.ann_file)
        remove_useless_info(self.coco)
        self.img_ids = self.coco.getImgIds()
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_names = tuple([c["name"] for c in self.coco.loadCats(self.cat_ids)])
        self.num_cats = len(self.cat_names)
        # self.classes = tuple([c["name"] for c in self.coco.loadCats(sorted(self.coco.getCatIds()))])
        # self.num_classes = len(self.classes)
        
        # self.imgs = None
        # # load annotations
        # for img_id in self.img_ids:
        #     img = self.coco.loadImgs(img_id)[0]
        #     self.file_names.append(img['file_name'])
        #     anns = self.coco.loadAnns(self.coco.getAnnIds(img_id))
            
        #     self.all_bboxes.append(bboxes)
        #     self.all_labels.append(labels)
        # self.num_samples = len(self.bboxes)

    def __getitem__(self,idx):
        img_id = self.img_ids[idx]
        img_meta = self.coco.loadImgs(img_id)[0]
        img = cv2.imread(osp.join(self.img_root, img_meta['file_name']))
        org_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        anns = self.coco.loadAnns(self.coco.getAnnIds(img_id))
        bboxes = np.array([
            [ann['bbox'][0], ann['bbox'][1], 
            ann['bbox'][0]+ann['bbox'][2], 
            ann['bbox'][1]+ann['bbox'][3],] for ann in anns])
        labels = np.array([self.cat_ids.index(ann['category_id']) + 1 for ann in anns])

        # draw_bboxes_on_img(org_img, boxes, labels)

        if self.train:
            transform = A.Compose([
                A.HorizontalFlip(),
                A.RandomScale(),
                A.Blur(),
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(),
                A.RandomResizedCrop(height=self.img_size[0], width=self.img_size[1], scale=(0.64, 1), ratio=(0.75, 1.3333)),
                A.Normalize(),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            transform = A.Compose([
                A.Resize(self.img_size[0], self.img_size[1]),
                A.Normalize(),
                ToTensorV2()
            ])

        transformed = transform(image=img, bboxes=bboxes, labels=labels)
        img = transformed['image']
        bboxes = transformed['bboxes']
        labels = transformed['labels']

        _, h, w = img.shape
        
        target = self.encoder(bboxes, labels, (w, h))# 7x7x30

        return img, target

    def __len__(self):
        return len(self.img_ids)

    def encoder(self, bboxes, labels, img_size):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]]
        labels (tensor) [...]
        return 7x7x30
        '''
        grid_num = 14
        w, h = img_size
        target = np.zeros((grid_num, grid_num, 2 * 5 + self.num_cats))

        if len(bboxes) == 0:
            return torch.FloatTensor(target)

        bboxes = np.array(bboxes) / np.array([w, h, w, h])

        cell_size = 1. / grid_num
        for bbox, label in zip(bboxes, labels):
            w_h = bbox[2:] - bbox[:2]
            cx_cy = (bbox[2:] + bbox[:2]) / 2
        # bbox_sizes = bboxes[:, 2:] - bboxes[:,:2]
        # bbox_centers = (bboxes[:,2:] + bboxes[:,:2]) / 2
        # for i in range(cxcy.size()[0]):
            # cxcy_sample = cxcy[i]
            ij = np.ceil((cx_cy / cell_size)) - 1 #
            target[int(ij[1]), int(ij[0]), 4] = 1
            target[int(ij[1]), int(ij[0]), 9] = 1
            target[int(ij[1]), int(ij[0]), int(label) + 9] = 1
            xy = ij * cell_size # 匹配到的网格的左上角相对坐标
            delta_xy = (cx_cy - xy) / cell_size
            target[int(ij[1]), int(ij[0]), 2:4] = w_h
            target[int(ij[1]), int(ij[0]), :2] = delta_xy
            target[int(ij[1]), int(ij[0]), 7:9] = w_h
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
        return torch.FloatTensor(target)
    # def BGR2RGB(self,img):
    #     return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    # def BGR2HSV(self,img):
    #     return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # def HSV2BGR(self,img):
    #     return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
    
    # def RandomBrightness(self,bgr):
    #     if random.random() < 0.5:
    #         hsv = self.BGR2HSV(bgr)
    #         h,s,v = cv2.split(hsv)
    #         adjust = random.choice([0.5,1.5])
    #         v = v*adjust
    #         v = np.clip(v, 0, 255).astype(hsv.dtype)
    #         hsv = cv2.merge((h,s,v))
    #         bgr = self.HSV2BGR(hsv)
    #     return bgr
    # def RandomSaturation(self,bgr):
    #     if random.random() < 0.5:
    #         hsv = self.BGR2HSV(bgr)
    #         h,s,v = cv2.split(hsv)
    #         adjust = random.choice([0.5,1.5])
    #         s = s*adjust
    #         s = np.clip(s, 0, 255).astype(hsv.dtype)
    #         hsv = cv2.merge((h,s,v))
    #         bgr = self.HSV2BGR(hsv)
    #     return bgr
    # def RandomHue(self,bgr):
    #     if random.random() < 0.5:
    #         hsv = self.BGR2HSV(bgr)
    #         h,s,v = cv2.split(hsv)
    #         adjust = random.choice([0.5,1.5])
    #         h = h*adjust
    #         h = np.clip(h, 0, 255).astype(hsv.dtype)
    #         hsv = cv2.merge((h,s,v))
    #         bgr = self.HSV2BGR(hsv)
    #     return bgr

    # def randomBlur(self,bgr):
    #     if random.random()<0.5:
    #         bgr = cv2.blur(bgr,(5,5))
    #     return bgr

    # def randomShift(self,bgr,boxes,labels):
    #     #平移变换
    #     center = (boxes[:,2:]+boxes[:,:2])/2
    #     if random.random() <0.5:
    #         height,width,c = bgr.shape
    #         after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
    #         after_shfit_image[:,:,:] = (104,117,123) #bgr
    #         shift_x = random.uniform(-width*0.2,width*0.2)
    #         shift_y = random.uniform(-height*0.2,height*0.2)
    #         #print(bgr.shape,shift_x,shift_y)
    #         #原图像的平移
    #         if shift_x>=0 and shift_y>=0:
    #             after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
    #         elif shift_x>=0 and shift_y<0:
    #             after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
    #         elif shift_x <0 and shift_y >=0:
    #             after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
    #         elif shift_x<0 and shift_y<0:
    #             after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

    #         shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
    #         center = center + shift_xy
    #         mask1 = (center[:,0] >0) & (center[:,0] < width)
    #         mask2 = (center[:,1] >0) & (center[:,1] < height)
    #         mask = (mask1 & mask2).view(-1,1)
    #         boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
    #         if len(boxes_in) == 0:
    #             return bgr,boxes,labels
    #         box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y)]]).expand_as(boxes_in)
    #         boxes_in = boxes_in+box_shift
    #         labels_in = labels[mask.view(-1)]
    #         return after_shfit_image,boxes_in,labels_in
    #     return bgr,boxes,labels

    # def randomScale(self,bgr,boxes):
    #     #固定住高度，以0.8-1.2伸缩宽度，做图像形变
    #     if random.random() < 0.5:
    #         scale = random.uniform(0.8,1.2)
    #         height,width,c = bgr.shape
    #         bgr = cv2.resize(bgr,(int(width*scale),height))
    #         scale_tensor = torch.FloatTensor([[scale,1,scale,1]]).expand_as(boxes)
    #         boxes = boxes * scale_tensor
    #         return bgr,boxes
    #     return bgr,boxes

    # def randomCrop(self,bgr,boxes,labels):
    #     if random.random() < 0.5:
    #         center = (boxes[:,2:]+boxes[:,:2])/2
    #         height,width,c = bgr.shape
    #         h = random.uniform(0.6*height,height)
    #         w = random.uniform(0.6*width,width)
    #         x = random.uniform(0,width-w)
    #         y = random.uniform(0,height-h)
    #         x,y,h,w = int(x),int(y),int(h),int(w)

    #         center = center - torch.FloatTensor([[x,y]]).expand_as(center)
    #         mask1 = (center[:,0]>0) & (center[:,0]<w)
    #         mask2 = (center[:,1]>0) & (center[:,1]<h)
    #         mask = (mask1 & mask2).view(-1,1)

    #         boxes_in = boxes[mask.expand_as(boxes)].view(-1,4)
    #         if(len(boxes_in)==0):
    #             return bgr,boxes,labels
    #         box_shift = torch.FloatTensor([[x,y,x,y]]).expand_as(boxes_in)

    #         boxes_in = boxes_in - box_shift
    #         boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
    #         boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
    #         boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
    #         boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

    #         labels_in = labels[mask.view(-1)]
    #         img_croped = bgr[y:y+h,x:x+w,:]
    #         return img_croped,boxes_in,labels_in
    #     return bgr,boxes,labels




    # def subMean(self,bgr,mean):
    #     mean = np.array(mean, dtype=np.float32)
    #     bgr = bgr - mean
    #     return bgr

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h,w,_ = im.shape
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
            return im_lr, boxes
        return im, boxes

    
    # def random_bright(self, im, delta=16):
    #     alpha = random.random()
    #     if alpha > 0.3:
    #         im = im * alpha + random.randrange(-delta,delta)
    #         im = im.clip(min=0,max=255).astype(np.uint8)
    #     return im

def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    coco_root = '/mnt/disk/Data/PublicDatasets/COCO/'
    train_dataset = yoloDataset(osp.join(coco_root, 'train2017'), ann_file=osp.join(coco_root, 'annotations/mini_instances_train2017.json'), train=True, transform=[transforms.ToTensor()], img_size=(448, 448))
    train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)
    train_iter = iter(train_loader)
    for i in range(100):
        img,target = next(train_iter)
        img = img.cpu().numpy()[0]
        target = target.cpu().numpy()
        print('img:', type(img).__name__, img.dtype, img.size())
        print('target:', type(target).__name__, target.dtype, target.size())


if __name__ == '__main__':
    main()


