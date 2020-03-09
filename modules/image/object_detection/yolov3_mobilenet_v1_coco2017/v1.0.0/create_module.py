# coding=utf-8
from paddlehub.module.module import create_module

create_module(
    directory='resources',
    name='yolov3_mobilenet_v1_coco2017',
    module_type='CV/object-detection',
    author='paddlepaddle',
    email='paddle-dev@baidu.com',
    summary=
    "Baidu's YOLOv3 model for object detection, with backbone MobileNet_V1.",
    version='1.0.0')
