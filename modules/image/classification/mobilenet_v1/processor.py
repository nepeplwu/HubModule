# coding=utf-8
import os
import numpy as np


def load_label_info(file_path):
    with open(file_path, 'r') as fr:
        text = fr.readlines()
        label_names = []
        for info in text:
            label_names.append(info.strip())
        return label_names
