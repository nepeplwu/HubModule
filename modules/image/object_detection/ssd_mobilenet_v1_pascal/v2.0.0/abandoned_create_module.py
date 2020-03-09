# coding=utf-8
from paddlehub.module.module import create_module

create_module(
    directory='resources',
    name='ssd_mobilenet_v1_pascal',
    module_type='CV/object-detection',
    author='paddlepaddle',
    email='paddle-dev@baidu.com',
    summary=
    "Baidu's SSD model for object detection, with backbone MobileNet_V1, trained in dataset Pascal VOC.",
    version='2.0.0')
