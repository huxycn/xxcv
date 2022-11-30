
为了方便个人用户训练模型，验证模型效果，想到了通过筛选特定类别来缩小COCO数据集的方法


前提跟其他代码库一样，准备好完整的COCO数据集，目录结构如下
```
COCO/
  annotations/
    instances_train2017.json
    instances_val2017.json
  train2017/
  val2017/
```

然后使用 mini_coco.py 脚本，筛选并生成新的标注文件，例如执行如下命令
```bash
python mini_coco.py /path/to/instances_train{val}2017.json 17,18
```
会在原标注文件同级目录下生成 mini_instances_train{val}2017.json 文件
其中 17,18 分别为 COCO 数据集中 cat,dog 的 id，可用筛选后的标注文件训练一个“猫狗”检测器

在 data/example 下有一份我自己生成的 mini(cat,dog) 标注文件的拷贝，我也用这份标注文件在 yolox 上训练了 nano 版模型（RTX3070，8h），配置文件如下


训练结果
| Model | mAP@0.5::0.95 (cat,dog) | log | weight |
| ---   | ---                     | --- | ---    |
| yolox_nano | 0.522              | train_log.txt | download |
