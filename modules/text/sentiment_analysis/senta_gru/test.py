# coding:utf-8
from __future__ import print_function

import json
import six

import paddlehub as hub

if __name__ == "__main__":
    # Load Senta_Bow Module
    senta = hub.Module(name='senta_gru')
    test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
    senta.context()
    results = senta.sentiment_classify(data={'text': test_text}, use_gpu=True)
    # execute predict and print the result
    for index, result in enumerate(results):
        if six.PY2:
            print(json.dumps(result, encoding="utf8", ensure_ascii=False))
        else:
            print(result)
    results = senta.sentiment_classify(texts=test_text, use_gpu=False)
    for index, result in enumerate(results):
        if six.PY2:
            print(json.dumps(result, encoding="utf8", ensure_ascii=False))
        else:
            print(result)
