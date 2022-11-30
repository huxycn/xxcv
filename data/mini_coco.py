import json
import time
import fire

from pathlib import Path


# https://github.com/Megvii-BaseDetection/YOLOX/blob/e80063b44a67f0f43c87152fb9d7623371286333/yolox/data/datasets/coco.py#L16
def remove_useless_info(dataset):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(dataset, dict):
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in dataset:
            for anno in dataset["annotations"]:
                anno.pop("segmentation", None)
# https://github.com/Megvii-BaseDetection/YOLOX/blob/e80063b44a67f0f43c87152fb9d7623371286333/yolox/data/datasets/coco.py#L32


def mini_coco(annotation_file, keep_ann_ids):
    keep_img_ids = set()
    
    # https://github.com/ppwwyyxx/cocoapi/blob/71e284ef862300e4319aacd523a64c7f24750178/PythonAPI/pycocotools/coco.py#L78
    if not annotation_file == None:
        print('loading annotations into memory...')
        tic = time.time()
        with open(annotation_file, 'r') as f:
            dataset = json.load(f)
        assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
    # https://github.com/ppwwyyxx/cocoapi/blob/71e284ef862300e4319aacd523a64c7f24750178/PythonAPI/pycocotools/coco.py#L84

    print('remove useless info')
    remove_useless_info(dataset)

    print(f'filter categories by keep_ann_ids: {keep_ann_ids}')
    if "categories" in dataset:
        dataset["categories"] = list(filter(lambda cat: cat["id"] in keep_ann_ids, dataset["categories"]))

    print(f'filter annotations by keep_ann_ids: {keep_ann_ids}')
    if "annotations" in dataset:
        dataset["annotations"] = list(filter(lambda ann: ann["category_id"] in keep_ann_ids, dataset["annotations"]))

        for anno in dataset["annotations"]:
            keep_img_ids.add(anno["image_id"])

    print(f'filter images by keep_img_ids: {len(keep_img_ids)}')
    if "images" in dataset:
        dataset["images"] = list(filter(lambda img: img["id"] in keep_img_ids, dataset["images"]))

    mini_annotation_file = Path(annotation_file).parent / ('mini_' + Path(annotation_file).name)
    print(f'dump mini annotations to file: {mini_annotation_file}')
    tic = time.time()
    with open(mini_annotation_file, 'w') as f:
        json.dump(dataset, f)
    print(f'Done (t={time.time() - tic:.2f}s)')


if __name__ == '__main__':
    """
    Example: python mini_coco.py /path/to/instances_train{val}2017.json 17,18
    """
    fire.Fire(mini_coco)
