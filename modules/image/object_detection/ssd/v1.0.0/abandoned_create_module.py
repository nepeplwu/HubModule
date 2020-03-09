# coding=utf-8
from paddlehub.module.module import create_module

create_module(
    directory='resources',
    name='ssd',
    module_type='CV/object-detection',
    author='paddlepaddle',
    email='paddle-dev@baidu.com',
    summary="Baidu's SSD Base Class.",
    version='1.0.0')
