import os
import os.path as osp
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

from yolo.yolo import YOLO
from yolo.yoloLoss import yoloLoss
from yolo.data import COCODetectionDataset

from yolo.utils.visualize import Visualizer
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from loguru import logger

use_gpu = torch.cuda.is_available()

file_root = '/mnt/disk/Data/PublicDatasets/VOC/all_imgs/'
learning_rate = 0.001
num_epochs = 50
batch_size = 12
use_resnet = True
net = YOLO('resnet18', feat_channels=512, S=7, B=2, C=2)
# net = YOLO('resnet50', feat_channels=2048)

logger.info(f'cuda: {torch.cuda.current_device()}, {torch.cuda.device_count()}')

criterion = yoloLoss(S=7, B=2, C=2, l_coord=5, l_noobj=0.5)
if use_gpu:
    net.cuda()

net.train()
# different learning rate
params=[]
params_dict = dict(net.named_parameters())
for key,value in params_dict.items():
    if key.startswith('features'):
        params += [{'params':[value],'lr':learning_rate*1}]
    else:
        params += [{'params':[value],'lr':learning_rate}]
optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

coco_root = '/mnt/disk/Data/PublicDatasets/COCO/'
train_dataset = COCODetectionDataset(osp.join(coco_root, 'train2017'), 
                            ann_file=osp.join(coco_root, 'annotations/mini_instances_train2017.json'), 
                            train=True, transform=[transforms.ToTensor()], 
                            img_size=(448, 448))
    
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=1)

test_dataset = COCODetectionDataset(osp.join(coco_root, 'val2017'), ann_file=osp.join(coco_root, 'annotations/mini_instances_val2017.json'), train=True, transform=[transforms.ToTensor()], img_size=(448, 448))
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=1)
logger.info(f'the dataset has {len(train_dataset)} images')
logger.info(f'the batch_size is {batch_size}')
logfile = open('log.txt', 'w')

num_iter = 0
# vis = Visualizer(env='main')
writer = SummaryWriter('tb_log')
best_test_loss = np.inf

for epoch in range(num_epochs):
    net.train()
    if epoch == 30:
        learning_rate=0.0001
    if epoch == 40:
        learning_rate=0.00001
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    logger.info(f'Starting epoch {epoch + 1}, {num_epochs}')
    logger.info(f'Learning Rate for this epoch: {learning_rate}')
    
    total_loss = 0.
    
    for i,(images,target) in enumerate(train_loader):
        images = Variable(images)
        target = Variable(target)
        if use_gpu:
            images,target = images.cuda(),target.cuda()
        
        pred = net(images)
        loss = criterion(pred,target)
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Iter [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}, average_loss: {total_loss / (i+1):.4f}')
            num_iter += 1
            writer.add_scalar('Train/loss', total_loss/(i+1), global_step=epoch*len(train_loader)+i+1)
            # vis.plot_train_val(loss_train=total_loss/(i+1))

    #validation
    validation_loss = 0.0
    net.eval()
    for i,(images,target) in enumerate(test_loader):
        # images = Variable(images,volatile=True)
        # target = Variable(target,volatile=True)
        if use_gpu:
            images,target = images.cuda(),target.cuda()
        
        with torch.no_grad():
            pred = net(images)
        loss = criterion(pred,target)
        logger.info(f'[{i}/{len(test_loader)}] val loss: {loss.item()}')
        validation_loss += loss.item()
    validation_loss /= len(test_loader)
    # vis.plot_train_val(loss_val=validation_loss)
    writer.add_scalar('Val/loss', validation_loss, global_step=epoch)
    
    if best_test_loss > validation_loss:
        best_test_loss = validation_loss
        logger.info(f'get best test loss {best_test_loss:.5f}')
        torch.save(net.state_dict(),'best.pth')
    logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')  
    logfile.flush()
    torch.save(net.state_dict(),'yolo.pth')
