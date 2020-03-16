#-*- coding:utf8 -*-
from __future__ import print_function

import json
import six

import paddlehub as hub

if __name__ == "__main__":
    # Load simnet_bow Module
    simnet_bow = hub.Module(name="simnet_bow")
    test_text_1 = ["这道题太难了", "这道题太难了", "这道题太难了"]
    test_text_2 = ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]

    test_text = [test_text_1, test_text_2]
    # execute predict and print the result
    results = simnet_bow.similarity(texts=test_text, use_gpu=True, batch_size=1)
    if six.PY2:
        print(json.dumps(results, encoding="utf8", ensure_ascii=False))
    else:
        print(results)

    max_score = -1
    result_text = ""
    for result in results:
        if result['similarity'] > max_score:
            max_score = result['similarity']
            result_text = result['text_2']

    print("The most matching with the %s is %s" % (test_text_1[0], result_text))

    input_dict = {'text_1': test_text_1, 'text_2': test_text_2}
    results = simnet_bow.similarity(data=input_dict, use_gpu=True, batch_size=2)
    if six.PY2:
        print(json.dumps(results, encoding="utf8", ensure_ascii=False))
    else:
        print(results)

    max_score = -1
    result_text = ""
    for result in results:
        if result['similarity'] > max_score:
            max_score = result['similarity']
            result_text = result['text_2']

    print("The most matching with the %s is %s" % (test_text_1[0], result_text))

    print(simnet_bow.get_vocab_path())
