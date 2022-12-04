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
        for cat in dataset["categories"]:
            cat.pop("supercategory", None)


class COCODetectionDataset(data.Dataset):
    def __init__(self, 
                 img_root, 
                 ann_file, 
                 train, 
                 transform=None,
                 img_size=(448, 448)):
        print('data init')
        self.img_root = img_root
        self.ann_file = ann_file
        self.train = train
        self.transform=transform
        self.img_size = img_size

        self.coco = COCO(self.ann_file)
        remove_useless_info(self.coco)
        self.img_ids = self.coco.getImgIds()
        self.cat_ids = self.coco.getCatIds()
        self.cat_id_names = self.coco.loadCats(self.cat_ids)
        # print(self.cat_id_names)
        # sys.exit()
        # self.cat_ids = sorted(self.coco.getCatIds())
        # self.cat_names = tuple([c["name"] for c in self.coco.loadCats(self.cat_ids)])
        self.num_cats = len(self.cat_ids)

    def __getitem__(self, idx):
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

        if self.transform is not None:
            transformed = self.transform(image=img, bboxes=bboxes, labels=labels)
            img = transformed['image']
            bboxes = transformed['bboxes']
            labels = transformed['labels']

        # return dict(img=img, bboxes=bboxes, labels=labels)

        _, h, w = img.shape
        
        target = self.encoder(bboxes, labels, (w, h))# 7x7x30

        return img, target

    def __len__(self):
        return len(self.img_ids)

    def encoder(self, bboxes, labels, img_size):
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


# def main():
#     from torch.utils.data import DataLoader
#     import torchvision.transforms as transforms
#     coco_root = '/mnt/disk/Data/PublicDatasets/COCO/'
#     train_dataset = yoloDataset(osp.join(coco_root, 'train2017'), ann_file=osp.join(coco_root, 'annotations/mini_instances_train2017.json'), train=True, transform=[transforms.ToTensor()], img_size=(448, 448))
#     train_loader = DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0)
#     train_iter = iter(train_loader)
#     for i in range(100):
#         img,target = next(train_iter)
#         img = img.cpu().numpy()[0]
#         target = target.cpu().numpy()
#         print('img:', type(img).__name__, img.dtype, img.size())
#         print('target:', type(target).__name__, target.dtype, target.size())


# if __name__ == '__main__':
#     main()


