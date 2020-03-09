# coding=utf-8
from paddlehub.module.module import create_module

create_module(
    directory='resources',
    name='faster_rcnn',
    module_type='CV/object-detection',
    author='paddlepaddle',
    email='paddle-dev@baidu.com',
    summary="Base Class of Baidu's Faster R-CNN model.",
    version='1.0.0')
