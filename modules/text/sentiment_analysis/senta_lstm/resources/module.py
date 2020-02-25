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
from paddlehub.io.parser import txt_parser
from paddlehub.module.module import runnable

from senta_lstm_python.net import lstm_net
from senta_lstm_python.processor import load_vocab, preprocess, postprocess


class DataFormatError(Exception):
    def __init__(self, *args):
        self.args = args


class SentaLSTM(hub.Module):
    def _initialize(self, user_dict=None):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "model")
        self.vocab_path = os.path.join(self.directory, "assets/vocab.txt")
        self.word_dict = load_vocab(self.vocab_path)
        self.lac = None

        cpu_config = AnalysisConfig(os.path.join(self.directory, "infer_model"))
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
            gpu_config = AnalysisConfig(
                os.path.join(self.directory, "infer_model"))
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(memory_pool_init_size_mb=500, device_id=0)
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

    def texts2tensor(self, texts, batch_size=2):
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

        if not self.lac:
            self.lac = hub.Module(name="lac")

        processed_results = preprocess(self.lac, predicted_data, self.word_dict)

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
            description="Run the lac module.",
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

        if args.user_dict:
            self.set_user_dict(args.user_dict)

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
        self.arg_config_group.add_argument(
            '--user_dict',
            type=str,
            default=None,
            help=
            "customized dictionary for intervening the word segmentation result"
        )

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
