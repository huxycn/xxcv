import cv2
import torch
import argparse

import albumentations as A

from albumentations import pytorch

from detectron2.config import LazyConfig, instantiate


CLASSES = ('cat', 'dog')
COLOR = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0]
]

def decoder(pred):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    grid_num = 14
    boxes=[]
    cls_indexs=[]
    probs = []
    cell_size = 1./grid_num
    pred = pred.data
    pred = pred.squeeze(0) #7x7x30
    contain1 = pred[:,:,4].unsqueeze(2)
    contain2 = pred[:,:,9].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2)
    mask1 = contain > 0.1 #大于阈值
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0)
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i,j,b] == 1:
                    #print(i,j,b)
                    box = pred[i,j,b*5:b*5+4]
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]])
                    xy = torch.FloatTensor([j,i])*cell_size #cell左上角  up left of cell
                    box[:2] = box[:2]*cell_size + xy # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())#转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    max_prob,cls_index = torch.max(pred[i,j,10:],0)
                    if float((contain_prob*max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1,4))
                        cls_indexs.append(cls_index.unsqueeze(0))
                        probs.append(contain_prob*max_prob)
    if len(boxes) ==0:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes,0) #(n,4)
        probs = torch.cat(probs,0) #(n,)
        cls_indexs = torch.cat(cls_indexs,0) #(n,)
    keep = nms(boxes,probs)
    return boxes[keep],cls_indexs[keep],probs[keep]

def nms(bboxes,scores,threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)
#
#start predict one image
#
def predict_gpu(model,image_name, transform, root_path=''):

    result = []
    img = cv2.imread(root_path+image_name)
    h, w, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transformed = transform(image=img)
    img = transformed['image']
    img = torch.unsqueeze(img, dim=0)

    img = img.cuda()

    with torch.no_grad():
        pred = model((img, None)) #1x7x7x30
    pred = pred.cpu()
    boxes,cls_indexs,probs =  decoder(pred)

    for i,box in enumerate(boxes):
        x1 = int(box[0]*w)
        x2 = int(box[2]*w)
        y1 = int(box[1]*h)
        y2 = int(box[3]*h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index) # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1,y1),(x2,y2),CLASSES[cls_index],image_name,prob])
    return result
        

def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model.eval()

    model.load_state_dict(torch.load('/mnt/disk/Code/yolo/pytorch-YOLO-v1/projects/yolo_v1/output/model_final.pth')['model'])

    transform_cfg = cfg.dataloader.test.dataset.transform
    transform_cfg.bbox_params = None
    transform = instantiate(transform_cfg)

    image = cv2.imread(args.image_file)
    # print('predicting...')
    result = predict_gpu(model, args.image_file, transform)
    for left_up,right_bottom,class_name,_,prob in result:
        color = COLOR[CLASSES.index(class_name)]
        cv2.rectangle(image,left_up,right_bottom,color,2)
        label = class_name+str(round(prob,2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1]- text_size[1])
        cv2.rectangle(image, (p1[0] - 2//2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, 8)

    cv2.imwrite('result.jpg',image)


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--image-file", default="", metavar="FILE", help="path to image file")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == '__main__':
    args = argument_parser().parse_args()
    main(args)
