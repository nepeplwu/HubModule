# -*- coding:utf-8 -*-
from __future__ import print_function

import json
import six

import paddlehub as hub

if __name__ == "__main__":
    # Load porn_detection module
    porn_detection_gru = hub.Module(name="porn_detection_gru")

    test_text = ["黄片下载", "打击黄牛党"]

    input_dict = {"text": test_text}
    results = porn_detection_gru.detection(
        data=input_dict, use_gpu=True, batch_size=1)

    print(results)
    results = porn_detection_gru.detection(
        texts=test_text, use_gpu=False, batch_size=2)
    print(results)

    print(porn_detection_gru.get_vocab_path())
    print(porn_detection_gru.get_labels())
