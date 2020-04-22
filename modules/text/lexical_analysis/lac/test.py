#-*- coding:utf8 -*-
from __future__ import print_function

import json
import six

import paddlehub as hub

if __name__ == "__main__":
    # Load LAC Module
    lac = hub.Module(name="lac", user_dict="user.dict")
    test_text = ["今天是个好日子", "天气预报说今天要下雨", "下一班地铁马上就要到了", "调料份量不能多，也不能少，味道才能正好"]

    # lac.set_user_dict("user.dict")
    results = lac.lexical_analysis(
        data={'text': test_text}, use_gpu=True, batch_size=1, return_tag=True)
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

    lac.del_user_dict()
    results = lac.lexical_analysis(
        texts=test_text, use_gpu=False, batch_size=10, return_tag=False)
    for result in results:
        if six.PY2:
            print(
                json.dumps(result['word'], encoding="utf8", ensure_ascii=False))
        else:
            print(result['word'])

    print(lac.get_tags())
