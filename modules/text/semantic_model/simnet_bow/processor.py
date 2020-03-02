# -*- coding: utf-8 -*-
import os
import io
import platform

import six
import paddle
import paddle.fluid as fluid
import numpy as np

import paddlehub as hub


def load_vocab(file_path):
    """
    load the given vocabulary
    """
    vocab = {}
    with io.open(file_path, 'r', encoding='utf8') as f:
        wid = 0
        for line in f:
            line = line.rstrip()
            parts = line.split('\t')
            vocab[parts[0]] = int(parts[1])
    vocab["<unk>"] = len(vocab)
    return vocab


text_a_key = "text_1"
text_b_key = "text_2"


def preprocess(lac, word_dict, data_dict):
    """
    Convert the word str to word id and pad the text
    """
    result = {text_a_key: [], text_b_key: []}
    processed_a = lac.lexical_analysis(data={'text': data_dict[text_a_key]})
    processed_b = lac.lexical_analysis(data={'text': data_dict[text_b_key]})
    for index, (text_a, text_b) in enumerate(zip(processed_a, processed_b)):
        result_i = {'processed': []}
        result_i['origin'] = data_dict[text_a_key][index]
        for word in text_a['word']:
            if word in word_dict:
                _index = word_dict[word]
            else:
                continue
            result_i['processed'].append(_index)
        result[text_a_key].append(result_i)

        result_i = {'processed': []}
        result_i['origin'] = data_dict[text_b_key][index]
        for word in text_b['word']:
            if word in word_dict:
                _index = word_dict[word]
            else:
                continue
            result_i['processed'].append(_index)
        result[text_b_key].append(result_i)
    return result


def postprocess(predict_out, data_info):
    """
    Convert model's output tensor to pornography label
    """
    result = []
    pred = predict_out.as_ndarray()
    for index in range(len(data_info[text_a_key])):
        result_i = {}
        result_i[text_a_key] = data_info[text_a_key][index]['origin']
        result_i[text_b_key] = data_info[text_b_key][index]['origin']
        result_i['similarity'] = float('%.4f' % ((pred[0][0] + 1) / 2))
        result.append(result_i)
    return result
