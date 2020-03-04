# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import json
import numpy as np
import os
import six

import paddle.fluid as fluid
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
import paddlehub as hub
from paddlehub.common.paddle_helper import get_variable_info
from paddlehub.common.utils import sys_stdin_encoding
from paddlehub.io.parser import txt_parser
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable
from paddlehub.module.module import serving
from paddlehub.reader import tokenization

from porn_detection_cnn.processor import load_vocab, preprocess, postprocess


class DataFormatError(Exception):
    def __init__(self, *args):
        self.args = args


@moduleinfo(
    name="porn_detection_cnn",
    version="1.1.0",
    summary="Baidu's open-source Porn Detection Model.",
    author="baidu-nlp",
    author_email="",
    type="nlp/sentiment_analysis")
class PornDetectionCNN(hub.Module):
    def _initialize(self):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "infer_model")
        self.tokenizer_vocab_path = os.path.join(self.directory, "assets",
                                                 "vocab.txt")
        self.vocab_path = os.path.join(self.directory, "assets",
                                       "word_dict.txt")
        self.vocab = load_vocab(self.vocab_path)
        self.sequence_max_len = 256
        self.tokenizer = tokenization.FullTokenizer(self.tokenizer_vocab_path)

        self.param_file = os.path.join(self.directory, "assets", "params.txt")

        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        cpu_config = AnalysisConfig(self.pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        self.cpu_predictor = create_paddle_predictor(cpu_config)

        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
            use_gpu = True
        except:
            use_gpu = False
        if use_gpu:
            gpu_config = AnalysisConfig(self.pretrained_model_path)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=500, device_id=0)
            self.gpu_predictor = create_paddle_predictor(gpu_config)

    def context(self, trainable=False):
        """
        Get the input ,output and program of the pretrained porn_detection_cnn
        Args:
             trainable(bool): whether fine-tune the pretrained parameters of porn_detection_cnn or not
        Returns:
             inputs(dict): the input variables of porn_detection_cnn (words)
             outputs(dict): the output variables of porn_detection_cnn (the sentiment prediction results)
             main_program(Program): the main_program of porn_detection_cnn with pretrained prameters
        """
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        program, feed_target_names, fetch_targets = fluid.io.load_inference_model(
            dirname=self.pretrained_model_path, executor=exe)

        with open(self.param_file, 'r') as file:
            params_list = file.readlines()
        for param in params_list:
            param = param.strip()
            var = program.global_block().var(param)
            var_info = get_variable_info(var)

            program.global_block().create_parameter(
                shape=var_info['shape'],
                dtype=var_info['dtype'],
                name=var_info['name'])

        for param in program.global_block().iter_parameters():
            param.trainable = trainable

        for name, var in program.global_block().vars.items():
            if name == feed_target_names[0]:
                inputs = {"words": var}
            # output of sencond layer from the end prediction layer (fc-softmax)
            if name == "@HUB_porn_detection_cnn@layer_norm_1.tmp_2":
                outputs = {
                    "class_probs": fetch_targets[0],
                    "sentence_feature": var
                }
        return inputs, outputs, program

    def texts2tensor(self, texts):
        """
        Tranform the texts(dict) to PaddleTensor
        Args:
             texts(dict): texts
        Returns:
             tensor(PaddleTensor): tensor with texts data
        """
        lod = [0]
        data = []
        for i, text in enumerate(texts):
            data += text['processed']
            lod.append(len(text['processed']) + lod[i])
        tensor = PaddleTensor(np.array(data).astype('int64'))
        tensor.name = "words"
        tensor.lod = [lod]
        tensor.shape = [lod[-1], 1]
        return tensor

    def to_unicode(self, texts):
        """
        Convert each element's type(str) of texts(list) to unicode in python2.7
        Args:
             texts(list): each element's type is str in python2.7
        Returns:
             texts(list): each element's type is unicode in python2.7
        """
        if six.PY2:
            unicode_texts = []
            for text in texts:
                if not isinstance(text, unicode):
                    unicode_texts.append(
                        text.decode(sys_stdin_encoding()).decode("utf8"))
                else:
                    unicode_texts.append(text)
            texts = unicode_texts
        return texts

    @serving
    def detection(self, texts=[], data={}, use_gpu=False, batch_size=1):
        """
        Get the porn prediction results results with the texts as input

        Args:
             texts(list): the input texts to be predicted, if texts not data
             data(dict): key must be 'text', value is the texts to be predicted, if data not texts
             use_gpu(bool): whether use gpu to predict or not
             batch_size(int): the program deals once with one batch

        Returns:
             results(dict): the porn prediction results
        """
        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
        except:
            use_gpu = False

        if texts != [] and isinstance(texts, list) and data == {}:
            predicted_data = texts
        elif texts == [] and isinstance(data, dict) and isinstance(
                data.get('text', None), list) and data['text']:
            predicted_data = data["text"]
        else:
            raise ValueError(
                "The input data is inconsistent with expectations.")

        predicted_data = self.to_unicode(predicted_data)
        processed_results = preprocess(predicted_data, self.tokenizer,
                                       self.vocab, self.sequence_max_len)
        tensor_words = self.texts2tensor(processed_results)
        if use_gpu:
            fetch_out = self.gpu_predictor.run([tensor_words])
        else:
            fetch_out = self.cpu_predictor.run([tensor_words])
        result = postprocess(fetch_out[0], processed_results)
        return result

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(
            description="Run the porn_detection_cnn module.",
            prog='hub run porn_detection_cnn',
            usage='%(prog)s',
            add_help=True)

        self.arg_input_group = self.parser.add_argument_group(
            title="Input options", description="Input data. Required")
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options",
            description=
            "Run configuration for controlling module behavior, not required.")

        self.add_module_config_arg()
        self.add_module_input_arg()

        args = self.parser.parse_args(argvs)

        try:
            input_data = self.check_input_data(args)
        except DataFormatError and RuntimeError:
            self.parser.print_help()
            return None

        results = self.sentiment_classify(
            texts=input_data, use_gpu=args.use_gpu, batch_size=args.batch_size)

        return results

    def add_module_config_arg(self):
        """
        Add the command config options
        """
        self.arg_config_group.add_argument(
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help="whether use GPU for prediction")

        self.arg_config_group.add_argument(
            '--batch_size',
            type=int,
            default=1,
            help="batch size for prediction")

    def add_module_input_arg(self):
        """
        Add the command input options
        """
        self.arg_input_group.add_argument(
            '--input_file',
            type=str,
            default=None,
            help="file contain input data")
        self.arg_input_group.add_argument(
            '--input_text', type=str, default=None, help="text to predict")

    def check_input_data(self, args):
        input_data = []
        if args.input_file:
            if not os.path.exists(args.input_file):
                print("File %s is not exist." % args.input_file)
                raise RuntimeError
            else:
                input_data = txt_parser.parse(args.input_file, use_strip=True)
        elif args.input_text:
            if args.input_text.strip() != '':
                if six.PY2:
                    input_data = [
                        args.input_text.decode(
                            sys_stdin_encoding()).decode("utf8")
                    ]
                else:
                    input_data = [args.input_text]
            else:
                print(
                    "ERROR: The input data is inconsistent with expectations.")

        if input_data == []:
            print("ERROR: The input data is inconsistent with expectations.")
            raise DataFormatError

        return input_data

    def get_vocab_path(self):
        """
        Get the path to the vocabulary whih was used to pretrain

        Returns:
             self.vocab_path(str): the path to vocabulary
        """
        return self.vocab_path


if __name__ == "__main__":
    porn_detection_cnn = PornDetectionCNN()
    test_text = ["黄片下载", "打击黄牛党"]

    results = porn_detection_cnn.detection(texts=test_text)
    for index, text in enumerate(test_text):
        results[index]["text"] = text
    for index, result in enumerate(results):
        if six.PY2:
            print(json.dumps(
                results[index], encoding="utf8", ensure_ascii=False))
        else:
            print(results[index])
    input_dict = {"text": test_text}
    results = porn_detection_cnn.detection(data=input_dict)
    for index, text in enumerate(test_text):
        results[index]["text"] = text
    for index, result in enumerate(results):
        if six.PY2:
            print(json.dumps(
                results[index], encoding="utf8", ensure_ascii=False))
        else:
            print(results[index])
