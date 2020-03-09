# coding=utf-8
from paddlehub.module.module import create_module

create_module(
    directory='resources',
    name='ssd_vgg16_512_coco2017',
    module_type='CV/object-detection',
    author='paddlepaddle',
    email='paddle-dev@baidu.com',
    summary=
    "Baidu's SSD model for object detection, with backbone VGG16, base_size 512.",
    version='1.0.0')
