# -*- coding:utf-8 -*-
from __future__ import print_function

import json
import six

import paddlehub as hub

if __name__ == "__main__":
    # Load emotion_detection_textcnn Module
    emotion_detection_textcnn = hub.Module(name='emotion_detection_textcnn')
    test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
    results = emotion_detection_textcnn.emotion_classify(
        data={'text': test_text}, use_gpu=True, batch_size=1)
    print(results)
    # execute predict and print the result
    results = emotion_detection_textcnn.emotion_classify(
        texts=test_text, use_gpu=False, batch_size=2)
    print(results)

    print(emotion_detection_textcnn.get_vocab_path())
    print(emotion_detection_textcnn.get_labels())
