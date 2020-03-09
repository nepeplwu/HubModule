# coding=utf-8
from paddlehub.module.module import create_module

create_module(
    directory='resources',
    name='faster_rcnn_resnet50_coco2017',
    module_type='CV/object-detection',
    author='paddlepaddle',
    email='paddle-dev@baidu.com',
    summary=
    "Baidu's Faster R-CNN model for object detection, with backbone ResNet50",
    version='1.0.0')
