import cv2
import pprint
from pathlib import Path

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tabulate import tabulate

from yolo.data import COCODetectionDataset
import torch
from torch.utils import data

from yolo.yoloLoss import yoloLoss


class YoloConfig:
    def __init__(self, coco_root):
        # ---------------- dataloader config ---------------- #
        self.input_size = [448, 448]

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        # self.mean = np.array([0., 0., 0.])
        # self.std = np.array([1., 1., 1.])
        self.name = 'yolo_v1'

        self.batch_size = 12
        self.num_workers = 4

        # ---------------- model config ---------------- #
        self.resnet_num_layers = 18

        # ---------------- train config ---------------- #
        self.device = 'cuda:0'

        self.lr = 0.001

        self.max_epoch = 50
        self.lr_steps = [30, 40]

        # ---------------- log & save config ---------------- #
        self.log_interval_step = 10
        self.eval_interval_epoch = 1
        self.save_interval_epoch = 10

        # self.name = 'lens_occlusion'
        # self.save_module_names = ['feature_extractor', 'lens_occlusion_seg_head']
        # self.data_root = Path(data_root) / self.name

        # train_set = LensOcclusionDataset(
        #     data_root=self.data_root,
        #     split='train',
        #     transform=self.get_transform(train=True)
        # )

        # self.train_loader = data.DataLoader(
        #     train_set,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=self.num_workers,
        #     pin_memory=False,
        #     drop_last=False
        # )

        # val_set = LensOcclusionDataset(
        #     data_root=self.data_root,
        #     split='test',
        #     transform=self.get_transform()
        # )

        # self.val_loader = data.DataLoader(
        #     val_set,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        #     pin_memory=False,
        #     drop_last=False
        # )
        
        train_set = COCODetectionDataset(
            Path(coco_root) / 'train2017', 
            ann_file=Path(coco_root) / 'annotations/mini_instances_train2017.json', 
            train=True, 
            transform=self.get_transform(train=True), 
            img_size=(448, 448)
        )
    
        self.train_loader = data.DataLoader(
            train_set, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4)

        val_set = COCODetectionDataset(
            Path(coco_root) / 'val2017', 
            ann_file=Path(coco_root) / 'annotations/mini_instances_val2017.json', 
            train=False, 
            transform=self.get_transform(train=False), 
            img_size=(448, 448)
        )
        self.val_loader = data.DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

        self.cat_id_names = train_set.cat_id_names
        self.cat_names = [cat['name'] for cat in self.cat_id_names]
        self.num_cats = len(self.cat_names)
        # self.num_classes = len(self.classes)
        self.num_train = len(train_set)
        self.num_val = len(val_set)

        # self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = yoloLoss(S=7, B=2, C=2, l_coord=5, l_noobj=0.5)

        # self.iou_metric = IoUMetric(self.classes)

    def get_transform(self, train=False):
        if train:
            transform = A.Compose([
                A.HorizontalFlip(),
                A.RandomScale(),
                A.Blur(),
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(),
                A.RandomResizedCrop(height=448, width=448, scale=(0.64, 1), ratio=(0.75, 1.3333)),
                A.Normalize(),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            transform = A.Compose([
                A.Resize(448, 448),
                A.Normalize(),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        return transform

    def denormalize(self, image_tensor):
        image_array = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        image_array = ((image_array * self.std + self.mean) * 255).astype(np.uint8)[:, :, ::-1]
        return image_array

    def get_optimizer_and_lr_scheduler(self, params):
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

        # optimizer = torch.optim.Adam(params, self.lr)
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.lr_steps)
        return optimizer, lr_scheduler

    @torch.no_grad()
    def eval(self, model):
        model.eval()

        # self.iou_metric.reset()

        losses = []
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = model(inputs)

            loss = self.criterion(outputs, targets)
            losses.append(loss.item())

        # eval_results = self.iou_metric.value()
        return None, np.mean(losses)

    @torch.no_grad()
    def inference_val(self, model):
        model.eval()

        for inputs, _ in self.val_loader:
            inputs = inputs.to(self.device)

            outputs = model.forward_train(inputs, path=self.name)
            _, predictions = torch.max(outputs.data, 1)

            for single_input, single_prediction in zip(inputs, predictions):
                input_image = self.denormalize(single_input)
                single_prediction = single_prediction.cpu().numpy().astype(np.uint8)
                predict_mask = single_prediction * 50
                yield input_image, predict_mask

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [
            (str(k), pprint.pformat(v))
            for k, v in vars(self).items()
            if not k.startswith("_")
        ]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")