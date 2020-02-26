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
from paddle.fluid.core import AnalysisConfig, create_paddle_predictor
import paddlehub as hub
from paddlehub.common.utils import sys_stdin_encoding
from paddlehub.io.parser import txt_parser
from paddlehub.module.module import moduleinfo
from paddlehub.module.module import runnable

import sys
sys.path.append("..")
from senta_lstm.net import lstm_net
from senta_lstm.processor import load_vocab, preprocess, postprocess


class DataFormatError(Exception):
    def __init__(self, *args):
        self.args = args


@moduleinfo(
    name="senta_lstm",
    version="1.1.0",
    summary="Baidu's open-source Sentiment Classification System.",
    author="baidu-nlp",
    author_email="paddle-dev@baidu.com",
    type="nlp/sentiment_analysis")
class SentaLSTM(hub.Module):
    def _initialize(self, user_dict=None):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "model")
        self.vocab_path = os.path.join(self.directory, "assets/vocab.txt")
        self.word_dict = load_vocab(self.vocab_path)
        self.lac = None

        self._set_config()

    def _set_config(self, ):
        """
        predictor config setting
        """
        cpu_config = AnalysisConfig(self.pretrained_model_path)
        cpu_config.disable_glog_info()
        cpu_config.disable_gpu()
        cpu_config.switch_use_feed_fetch_ops(False)
        cpu_config.switch_ir_optim(True)
        cpu_config.enable_memory_optim()
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
            gpu_config.switch_use_feed_fetch_ops(False)
            gpu_config.switch_ir_optim(True)
            gpu_config.enable_memory_optim()
            self.gpu_predictor = create_paddle_predictor(gpu_config)

    def context(
            self,
            trainable=False,
    ):
        """
        Get the input ,output and program of the pretrained senta_lstm

        Args:
             trainable(bool): whether fine-tune the pretrained parameters of senta_lstm or not

        Returns:
             inputs(dict): the input variables of senta_lstm (words)
             outputs(dict): the output variables of senta_lstm (the sentiment prediction results)
             main_program(Program): the main_program of lac with pretrained prameters
        """
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            with fluid.unique_name.guard("@HUB_senta_lstm@"):
                data = fluid.layers.data(
                    name="words", shape=[1], dtype="int64", lod_level=1)
                label = fluid.layers.data(
                    name="label", shape=[1], dtype="int64")

                cost, acc, pred, fc = lstm_net(data, label, 1256606)

                for param in main_program.global_block().iter_parameters():
                    param.trainable = trainable

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)

                # load the senta_lstm pretrained model
                def if_exist(var):
                    return os.path.exists(
                        os.path.join(self.pretrained_model_path, var.name))

                fluid.io.load_vars(
                    exe, self.pretrained_model_path, predicate=if_exist)

                inputs = {"words": data}
                outputs = {"class_probs": pred, "sentence_feature": fc}

                return inputs, outputs, main_program

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

    def sentiment_classify(self, texts=[], data={}, use_gpu=False,
                           batch_size=1):
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

        if texts != [] and isinstance(texts, list) and data == {}:
            predicted_data = texts
        elif texts == [] and isinstance(data, dict) and isinstance(
                data.get('text', None), list) and data['text']:
            predicted_data = data["text"]
        else:
            raise ValueError(
                "The input data is inconsistent with expectations.")

        predicted_data = self.to_unicode(predicted_data)
        if not self.lac:
            self.lac = hub.Module(
                directory="/ssd2/home/zhangxuefei/.paddlehub/modules/lac")
        processed_results = preprocess(self.lac, predicted_data, self.word_dict,
                                       use_gpu)

        lod = [0]
        data = []
        for i, text in enumerate(processed_results):
            data += text['processed']
            lod.append(len(text['processed']) + lod[i])

        if use_gpu:
            names = self.gpu_predictor.get_input_names()
            input_tensor = self.gpu_predictor.get_input_tensor(names[0])
            input_tensor.reshape([lod[-1], 1])
            input_tensor.copy_from_cpu(
                np.array(data).reshape([lod[-1], 1]).astype("int64"))
            input_tensor.set_lod([lod])
            self.gpu_predictor.zero_copy_run()
            output_name = self.gpu_predictor.get_output_names()
            output_tensor = self.gpu_predictor.get_output_tensor(output_name[0])
        else:
            names = self.cpu_predictor.get_input_names()
            input_tensor = self.cpu_predictor.get_input_tensor(names[0])
            input_tensor.reshape([lod[-1], 1])
            input_tensor.copy_from_cpu(
                np.array(data).reshape([lod[-1], 1]).astype("int64"))
            input_tensor.set_lod([lod])
            self.cpu_predictor.zero_copy_run()
            output_name = self.cpu_predictor.get_output_names()
            output_tensor = self.cpu_predictor.get_output_tensor(output_name[0])

        predict_out = output_tensor.copy_to_cpu()
        result = postprocess(predict_out, processed_results)
        return result

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(
            description="Run the senta_lstm module.",
            prog='hub run senta_lstm',
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
        if six.PY2:
            try:
                results = json.dumps(
                    results, encoding="utf8", ensure_ascii=False)
            except:
                pass

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

    def get_vocab_path(self, ):
        """
        Get the path to the vocabulary whih was used to pretrain

        Returns:
             self.vocab_path(str): the path to vocabulary
        """
        return self.vocab_path


if __name__ == "__main__":
    senta = SentaLSTM()
    # Data to be predicted
    test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]

    # execute predict and print the result
    input_dict = {"text": test_text}
    results = senta.sentiment_classify(data=input_dict)

    for index, result in enumerate(results):
        if six.PY2:
            print(json.dumps(
                results[index], encoding="utf8", ensure_ascii=False))
        else:
            print(results[index])
