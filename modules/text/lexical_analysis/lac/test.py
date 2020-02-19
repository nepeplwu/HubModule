# coding:utf-8
from __future__ import print_function

import json
import os
import six
import time

import paddlehub as hub

if __name__ == "__main__":
    # Load LAC Module
    lac = hub.Module(name="lac")
    test_text = ["今天是个好日子", "天气预报说今天要下雨", "下一班地铁马上就要到了", "调料份量不能多，也不能少，味道才能正好"]
    lac.set_user_dict("user.dict")
    results = lac.lexical_analysis(data=["今天是个好日子"], use_gpu=True, batch_size=1)
    # execute predict and print the result
    for result in results:
        if six.PY2:
            print(
                json.dumps(result['word'], encoding="utf8", ensure_ascii=False))
            print(
                json.dumps(result['tag'], encoding="utf8", ensure_ascii=False))
        else:
            print(result['word'])
            print(result['tag'])
    results = lac.lexical_analysis(data=test_text, use_gpu=False, batch_size=10)
    for result in results:
        if six.PY2:
            print(
                json.dumps(result['word'], encoding="utf8", ensure_ascii=False))
            print(
                json.dumps(result['tag'], encoding="utf8", ensure_ascii=False))
        else:
            print(result['word'])
            print(result['tag'])
