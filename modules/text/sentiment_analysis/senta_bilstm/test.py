# -*- coding:utf-8 -*-
from __future__ import print_function

import json
import six

import paddlehub as hub

if __name__ == "__main__":
    # Load Senta Module
    senta = hub.Module(name='senta_bilstm')
    test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
    senta.context()
    results = senta.sentiment_classify(
        data={'text': test_text}, use_gpu=True, batch_size=1)
    print(results)
    # execute predict and print the result
    results = senta.sentiment_classify(
        texts=test_text, use_gpu=False, batch_size=2)
    print(results)

    print(senta.get_vocab_path())
    print(senta.get_labels())
