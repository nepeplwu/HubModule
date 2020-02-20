# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import io
import json
import numpy as np
import os
import six

import paddle
import paddle.fluid as fluid
from paddle.fluid.core import PaddleDType, PaddleTensor, AnalysisConfig, create_paddle_predictor
import paddlehub as hub
from paddlehub.common.logger import logger
from paddlehub.io.parser import txt_parser
from paddlehub.module.module import runnable

from network import lex_net
from processor import Interventer, load_kv_dict, word_to_ids, parse_result


class DataFormatError(Exception):
    def __init__(self, *args):
        self.args = args


class LAC(hub.Module):
    def _initialize(self, user_dict=None):
        """
        initialize with the necessary elements
        """
        self.pretrained_model_path = os.path.join(self.directory, "infer_model")
        self.word2id_dict = load_kv_dict(
            os.path.join(self.directory, "asserts/word.dic"),
            reverse=True,
            value_func=int)
        self.id2word_dict = load_kv_dict(
            os.path.join(self.directory, "asserts/word.dic"))
        self.label2id_dict = load_kv_dict(
            os.path.join(self.directory, "asserts/tag.dic"),
            reverse=True,
            value_func=int)
        self.id2label_dict = load_kv_dict(
            os.path.join(self.directory, "asserts/tag.dic"))
        self.word_replace_dict = load_kv_dict(
            os.path.join(self.directory, "asserts/q2b.dic"))
        self.unigram_dict_path = os.path.join(self.directory,
                                              "asserts/unigram.dict")
        self.oov_id = self.word2id_dict['OOV']
        self.word_dict_len = max(map(int, self.word2id_dict.values())) + 1
        self.label_dict_len = max(map(int, self.label2id_dict.values())) + 1
        self.tag_file = os.path.join(self.directory, "asserts/tag_file.txt")

        if user_dict:
            self.set_user_dict(dict_path=user_dict)
        else:
            self.interventer = None

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

    def context(
            self,
            trainable=False,
    ):
        """
        Get the input ,output and program of the pretrained lac

        Args:
             trainable(bool): whether fine-tune the pretrained parameters of lac or not

        Returns:
             inputs(dict): the input variables of lac (words)
             outputs(dict): the output variables of lac (the word segmentation results)
             main_program(Program): the main_program of lac with pretrained prameters
        """
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            with fluid.unique_name.guard():
                crf_decode, word = lex_net(self.word_dict_len,
                                           self.label_dict_len)

                for param in main_program.global_block().iter_parameters():
                    param.trainable = trainable

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)

                # load the lac pretrained model
                def if_exist(var):
                    return os.path.exists(
                        os.path.join(self.pretrained_model_path, var.name))

                fluid.io.load_vars(
                    exe, self.pretrained_model_path, predicate=if_exist)

                inputs = {"words": word}
                outputs = {"predicted": crf_decode}

                return inputs, outputs, main_program

    def set_user_dict(self, dict_path):
        """
        Set the costomized dictionary if you wanna exploit the self-defined dictionary

        Args:
             dict_path(str): the directory to the costomized dictionary
        """
        if not os.path.exists(dict_path):
            raise RuntimeError("File %s is not exist." % dict_path)
        self.interventer = Interventer(self.unigram_dict_path, dict_path)

    def del_user_dict(self, ):
        """
        Delete the costomized dictionary if you don't wanna exploit the self-defined dictionary any longer
        """

        if self.interventer:
            self.interventer = None
            print("Successfully delete the customized dictionary!")

    def texts2tensor(self, texts):
        """
        Tranform the texts(list) to PaddleTensor

        Args:
             texts(list): texts

        Returns:
             tensor(PaddleTensor): tensor with texts data
        """
        lod = [0]
        data = []
        for i, text in enumerate(texts):
            text_inds = word_to_ids(
                text,
                self.word2id_dict,
                self.word_replace_dict,
                oov_id=self.oov_id)
            data += text_inds
            lod.append(len(text_inds) + lod[i])

        tensor = PaddleTensor(np.array(data).astype('int64'))
        tensor.name = "words"
        tensor.lod = [lod]
        tensor.shape = [lod[-1], 1]
        return tensor

    def lexical_analysis(self,
                         texts=[],
                         data={},
                         use_gpu=False,
                         batch_size=1,
                         user_dict=None):
        """
        Get the word segmentation results with the texts as input

        Args:
             texts(list): the input texts to be segmented, if texts not data
             data(dict): key must be 'text', value is the texts to be segmented, if data not texts
             use_gpu(bool): whether use gpu to predict or not
             batch_size(int): the program deals once with one batch
             user_dict(None): the parameter is not to be recommended. Please set the dictionause the function set_user_dict()

        Returns:
             results(dict): the word segmentation results
        """
        if user_dict:
            logger.warning(
                "If you wanna use customized dictionary, please set the dictionause the function set_user_dict(). The parameter user_dict has been dropped!"
            )

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

        tensor_words = self.texts2tensor(predicted_data)
        if use_gpu:
            crf_decode = self.gpu_predictor.run([tensor_words])
        else:
            crf_decode = self.cpu_predictor.run([tensor_words])
        result = parse_result(
            predicted_data,
            crf_decode[0],
            self.id2label_dict,
            interventer=self.interventer)
        return result

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command
        """
        self.parser = argparse.ArgumentParser(
            description="Run the lac module.",
            prog='hub run lac',
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

        results = self.lexical_analysis(
            texts=input_data, use_gpu=args.use_gpu, batch_size=args.batch_size)
        if six.PY2:
            try:
                results = json.dumps(
                    results, encoding="utf8", ensure_ascii=False)
            except:
                pass

        return results

    def get_tags(self, ):
        """
        Get the lac tags

        Returns:
             self.tag_name_dict(dict):lac tags
        """
        self.tag_name_dict = {}
        with io.open(self.tag_file, encoding="utf8") as f:
            for line in f:
                tag, tag_name = line.strip().split(" ")
                self.tag_name_dict[tag] = tag_name
        return self.tag_name_dict

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


if __name__ == '__main__':
    lac = LAC()
    test_text = ["今天是个好日子", "天气预报说今天要下雨", "下一班地铁马上就要到了", "调料份量不能多，也不能少，味道才能正好"]

    # lac.set_user_dict("user.dict")
    input_dict = {'text': ["今天是个好日子"]}
    results = lac.lexical_analysis(data=test_text, use_gpu=True, batch_size=1)
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
