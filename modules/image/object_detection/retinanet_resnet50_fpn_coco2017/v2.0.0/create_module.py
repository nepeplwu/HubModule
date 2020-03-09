# coding=utf-8
from paddlehub.module.module import create_module

create_module(
    directory='resources',
    name='retinanet_resnet50_fpn_coco2017',
    module_type='CV/object-detection',
    author='paddlepaddle',
    email='paddle-dev@baidu.com',
    summary=
    "Baidu's RetinaNet model for object detection, with backbone ResNet50 and FPN.",
    version='2.0.0')
