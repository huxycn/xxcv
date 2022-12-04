

import torch
from omegaconf import OmegaConf

import albumentations as A
from albumentations import pytorch
from yolo_v1.dataset.coco import COCODetectionDataset
from yolo_v1.modules.yolo import YOLO
from yolo_v1.modules.resnet import resnet18
from yolo_v1.modules.loss import YOLOLoss

from detectron2.solver.build import get_default_optimizer_params

from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.solver import WarmupParamScheduler

import torch
from omegaconf import OmegaConf
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.solver import WarmupParamScheduler
from detectron2.solver.build import get_default_optimizer_params
from detectron2.config import LazyCall as L
from detectron2.model_zoo import get_config
from detectron2.data.samplers import TrainingSampler, InferenceSampler
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm


def build_data_loader(dataset, batch_size, num_workers, training=True):
    return torch.utils.data.DataLoader(
        dataset,
        sampler=(TrainingSampler if training else InferenceSampler)(len(dataset)),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

dataloader = OmegaConf.create()
dataloader.train = L(build_data_loader)(
    dataset=L(COCODetectionDataset)(
        img_root='/mnt/disk/Data/PublicDatasets/COCO/train2017',
        ann_file='/mnt/disk/Data/PublicDatasets/COCO/annotations/mini_instances_train2017.json',
        train=True,
        transform = L(A.Compose)(
            transforms=[
                L(A.HorizontalFlip)(),
                L(A.RandomScale)(),
                L(A.Blur)(),
                L(A.RandomBrightnessContrast)(),
                L(A.HueSaturationValue)(),
                L(A.RandomResizedCrop)(height=448, width=448, scale=(0.64, 1), ratio=(0.75, 1.3333)),
                L(A.Normalize)(),
                L(pytorch.ToTensorV2)()
            ], 
            bbox_params=L(A.BboxParams)(
                format='pascal_voc', 
                label_fields=['labels']
            )
        ),
        img_size=(448, 448)
    ),
    batch_size=12,
    num_workers=4,
    training=True
)

dataloader.test = L(build_data_loader)(
    dataset=L(COCODetectionDataset)(
        img_root='/mnt/disk/Data/PublicDatasets/COCO/val2017', 
        ann_file='/mnt/disk/Data/PublicDatasets/COCO/annotations/mini_instances_val2017.json', 
        train=False, 
        transform = L(A.Compose)(
            transforms = [
                L(A.Resize)(height=448, width=448),
                L(A.Normalize)(),
                L(pytorch.ToTensorV2)()
            ], 
            bbox_params = L(A.BboxParams)(
                format = 'pascal_voc', 
                label_fields = ['labels']
            )
        ),
        img_size=(448, 448)
    ),
    batch_size=12,
    num_workers=4,
    training=False
)

model = L(YOLO)(
    backbone=L(resnet18)(pretrained=True), 
    feat_channels=512, 
    S=7, B=2, C=2, 
    criterion=L(YOLOLoss)(
        S=7, B=2, C=2, l_coord=5, l_noobj=0.5))

optimizer = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(), 
    lr=0.01, 
    momentum=0.9, 
    weight_decay=5e-4
)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01, 0.001], milestones=[30, 60, 90, 100]
    ),
    warmup_length=1 / 100,
    warmup_factor=0.1,
)

train = get_config("common/train.py").train
train.init_checkpoint = None
train.max_iter = 25 * 700
