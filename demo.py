from hutils import cv_utils as cv

import numpy as np
import albumentations as A

def draw_bboxes_on_img(_img, _bboxes, _labels):
    for bbox, label in zip(_bboxes, _labels):
        print(bbox, label)
        cv.draw_bbox2d(_img, bbox[:2], bbox[2:], cv.RED, label=f'{label}')


img = cv.imread('demo/dog.jpg')
org_img = img.copy()
bboxes = [[100, 100, 200, 300]]
labels = [1]

draw_bboxes_on_img(org_img, bboxes, labels)

transform = A.Compose([
    A.VerticalFlip(p=1)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

transformed = transform(image=img, bboxes=bboxes, labels=labels)

t_img = transformed['image']
t_bboxes = transformed['bboxes']
t_labels = transformed['labels']

draw_bboxes_on_img(t_img, t_bboxes, t_labels)

show_img = np.vstack([org_img, t_img])
