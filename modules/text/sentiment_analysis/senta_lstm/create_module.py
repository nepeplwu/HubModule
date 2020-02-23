# -*- coding:utf-8 -*-
from paddlehub.module.module import create_module

create_module(
    directory='resources',
    name='senta_lstm',
    module_type='nlp/sentiment_analysis',
    author='baidu-nlp ',
    email='nlp@baidu.com ',
    summary=
    "Baidu's open-source lexical analysis tool for Chinese, including word segmentation, part-of-speech tagging & named entity recognition",
    version='1.1.0')
