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
from paddle.fluid.core import PaddleDType, PaddleTensor, AnalysisConfig, create_paddle_predictor
import paddlehub as hub
from paddlehub.common.utils import sys_stdin_encoding
from paddlehub.io.parser import txt_parser
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable

import sys
sys.path.append("..")

from simnet_bow.processor import load_vocab, preprocess, postprocess

import time

dtype_map = {
    fluid.core.VarDesc.VarType.FP32: "float32",
    fluid.core.VarDesc.VarType.FP64: "float64",
    fluid.core.VarDesc.VarType.FP16: "float16",
    fluid.core.VarDesc.VarType.INT32: "int32",
    fluid.core.VarDesc.VarType.INT16: "int16",
    fluid.core.VarDesc.VarType.INT64: "int64",
    fluid.core.VarDesc.VarType.BOOL: "bool",
    fluid.core.VarDesc.VarType.INT16: "int16",
    fluid.core.VarDesc.VarType.UINT8: "uint8",
    fluid.core.VarDesc.VarType.INT8: "int8",
}


def convert_dtype_to_string(dtype):
    if dtype in dtype_map:
        return dtype_map[dtype]
    raise TypeError("dtype shoule in %s" % list(dtype_map.keys()))


class DataFormatError(Exception):
    def __init__(self, *args):
        self.args = args


@moduleinfo(
    name="simnet_bow",
    version="1.1.0",
    summary=
    " Baidu's open-source similarity network model based on bow_pairwisei.",
    author="baidu-nlp",
    author_email="paddle-dev@baidu.com",
    type="nlp/sentiment_analysis")
class SimnetBow(hub.Module):
    def _initialize(self, ):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "model")
        self.vocab_path = os.path.join(self.directory, "assets/vocab.txt")
        self.vocab = load_vocab(self.vocab_path)
        self.lac = None

        self._set_config()

    def _set_config(self, ):
        """
        predictor config setting
        """
        cpu_config = AnalysisConfig(self.pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        cpu_config.switch_ir_optim(False)
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

    def context(
            self,
            trainable=False,
    ):
        """
        Get the input ,output and program of the pretrained simnet_bow
        Args:
             trainable(bool): whether fine-tune the pretrained parameters of senta_bilstm or not
        Returns:
             inputs(dict): the input variables of senta_bilstm (words)
             outputs(dict): the output variables of senta_bilstm (the sentiment prediction results)
             main_program(Program): the main_program of lac with pretrained prameters
        """
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        program, feed_var_names, fetch_targets = fluid.io.load_inference_model(
            'model', exe)
        with open("assets/params.txt") as file:
            params_list = file.read().split("\n")

        for param in params_list:
            var = program.global_block().var(param)
            var_info = {
                'name': var.name,
                'dtype': convert_dtype_to_string(var.dtype),
                'lod_level': var.lod_level,
                'shape': var.shape,
                'stop_gradient': var.stop_gradient,
                'is_data': var.is_data,
                'error_clip': var.error_clip
            }
            program.global_block().create_parameter(**var_info)

        for param in program.global_block().iter_parameters():
            param.trainable = trainable

        text_a = program.global_block().vars[feed_var_names[0]]
        text_b = program.global_block().vars[feed_var_names[0]]
        inputs = {"text_a": text_a, "text_b": text_b}
        outputs = {
            "similarity": fetch_targets[1],
            "sentences_feature": fetch_targets[0]
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

    def similarity(self, data={}, use_gpu=False, batch_size=1):
        """
        Get the sentiment prediction results results with the texts as input
        Args:
             texts(list): the input texts to be predicted, if texts not data
             data(dict): key must be 'text', value is the texts to be predicted, if data not texts
             use_gpu(bool): whether use gpu to predict or not
             batch_size(int): the program deals once with one batch
        Returns:
             results(dict): the word segmentation results
        """
        try:
            _places = os.environ["CUDA_VISIBLE_DEVICES"]
            int(_places[0])
        except:
            use_gpu = False

        data['text_1'] = self.to_unicode(data['text_1'])
        data['text_2'] = self.to_unicode(data['text_2'])
        if not self.lac:
            self.lac = hub.Module(
                directory="/ssd2/home/zhangxuefei/.paddlehub/modules/lac")
        processed_results = preprocess(self.lac, self.vocab, data, use_gpu)

        tensor_words_1 = self.texts2tensor(processed_results["text_1"])
        tensor_words_2 = self.texts2tensor(processed_results["text_2"])

        if use_gpu:
            fetch_out = self.gpu_predictor.run([tensor_words_1, tensor_words_2])
        else:
            fetch_out = self.cpu_predictor.run([tensor_words_1, tensor_words_2])
        result = postprocess(fetch_out[1], processed_results)
        return result

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(
            description="Run the simnet_bow module.",
            prog='hub run simnet_bow',
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

        results = self.similarity(
            data=input_data, use_gpu=args.use_gpu, batch_size=args.batch_size)

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
            '--text_1', type=str, default=None, help="text to predict")
        self.arg_input_group.add_argument(
            '--text_2', type=str, default=None, help="text to predict")

    def check_input_data(self, args):
        input_data = {}
        if args.input_file:
            if not os.path.exists(args.input_file):
                print("File %s is not exist." % args.input_file)
                raise RuntimeError
            else:
                input_data = txt_parser.parse(args.input_file, use_strip=True)
        elif args.text_1 and args.text_2:
            if args.text_1.strip() != '' and args.text_2.strip() != '':
                if six.PY2:
                    input_data = {
                        "text_1":
                        args.text_1.strip().decode(
                            sys_stdin_encoding()).decode("utf8"),
                        "text_2":
                        args.text_2.strip().decode(
                            sys_stdin_encoding()).decode("utf8")
                    }
                else:
                    input_data = {"text_1": args.text_1, "text_2": args.text_2}
            else:
                print(
                    "ERROR: The input data is inconsistent with expectations.")

        if input_data == {}:
            print("ERROR: The input data is inconsistent with expectations.")
            raise DataFormatError

        return input_data

    def get_vocab_path(self, ):
        """
        Get the path to the vocabulary whih was used to pretrain
        Returns:
             self.vocab_path(str): the path to vocabulary
        """
        return self.vocab_path


if __name__ == "__main__":

    simnet_bow = SimnetBow()
    # Data to be predicted
    test_text_1 = ["这道题太难了", "这道题太难了", "这道题太难了"]
    test_text_2 = ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]

    inputs = {"text_1": test_text_1, "text_2": test_text_2}
    results = simnet_bow.similarity(data=inputs)

    max_score = -1
    result_text = ""
    for result in results:
        if result['similarity'] > max_score:
            max_score = result['similarity']
            result_text = result['text_2']

    print("The most matching with the %s is %s" % (test_text_1[0], result_text))
