# coding=utf-8
from paddlehub.module.module import create_module

create_module(
    directory='resources',
    name='yolov3',
    module_type='CV/object-detection',
    author='paddlepaddle',
    email='paddle-dev@baidu.com',
    summary="Baidu's YOLOv3 Base Class.",
    version='1.0.0')
