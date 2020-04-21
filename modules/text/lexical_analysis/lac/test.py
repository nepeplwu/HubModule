#-*- coding:utf8 -*-
from __future__ import print_function

import json
import six

import paddlehub as hub

if __name__ == "__main__":
    # Load LAC Module
    lac = hub.Module(name="lac")
    test_text = ["今天是个好日子", "调料份量不能多，也不能少，味道才能正好"]

    # lac.set_user_dict("user.dict")
    results = lac.lexical_analysis(
        data={'text': test_text}, use_gpu=True, batch_size=1, return_tag=True)
    print(results)

    # lac.del_user_dict()
    lac.set_user_dict('user.dict')
    results = lac.lexical_analysis(
        texts=test_text, use_gpu=False, batch_size=10, return_tag=False)
    print(results)
    print(lac.get_tags())
