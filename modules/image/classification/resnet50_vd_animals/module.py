# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import ast
import argparse
import os

import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import moduleinfo, runnable, serving

from resnet50_vd_animals.processor import postprocess
from resnet50_vd_animals.data_feed import reader
from resnet50_vd_animals.resnet_vd import ResNet50_vd


@moduleinfo(
    name="resnet50_vd_animals",
    type="CV/image_classification",
    author="baidu-vis",
    author_email="",
    summary=
    "ResNet50vd is a image classfication model, this module is trained with Baidu self-built animals dataset.",
    version="1.0.0")
class ResNet50vdAnimals(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(
            self.directory, "model")
        with open(
                os.path.join(self.directory, "label_list.txt"),
                "r",
                encoding='utf-8') as file:
            content = file.read()
        self.label_list = content.split("\n")
        self._set_config()

    def _set_config(self):
        """
        predictor config setting
        """
        cpu_config = AnalysisConfig(self.default_pretrained_model_path)
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
            gpu_config = AnalysisConfig(self.default_pretrained_model_path)
            gpu_config.disable_glog_info()
            gpu_config.enable_use_gpu(
                memory_pool_init_size_mb=1000, device_id=0)
            self.gpu_predictor = create_paddle_predictor(gpu_config)

    def context(self, trainable=True, pretrained=True):
        """context for transfer learning.

        Args:
            trainable (bool): Set parameters in program to be trainable.
            pretrained (bool) : Whether to load pretrained model.

        Returns:
            inputs (dict): key is 'image', corresponding vaule is image tensor.
            ouputs (dict): key is 'classification', corresponding value is the result of classification.
            context_prog (fluid.Program): program for transfer learning.
        """
        context_prog = fluid.Program()
        startup_prog = fluid.Program()
        with fluid.program_guard(context_prog, startup_prog):
            with fluid.unique_name.guard():
                image = fluid.layers.data(
                    name="image", shape=[3, 224, 224], dtype="float32")
                resnet_vd = ResNet50_vd()
                ouput = resnet_vd.net(input=image, class_dim=7979)
                inputs = {'image': image}
                ouputs = {'classification': ouput}

                place = fluid.CPUPlace()
                exe = fluid.Executor(place)
                # pretrained
                if pretrained:

                    def _if_exist(var):
                        b = os.path.exists(
                            os.path.join(self.default_pretrained_model_path,
                                         var.name))
                        return b

                    fluid.io.load_vars(
                        exe,
                        self.default_pretrained_model_path,
                        context_prog,
                        predicate=_if_exist)
                else:
                    exe.run(startup_prog)
                # trainable
                for param in context_prog.global_block().iter_parameters():
                    param.trainable = trainable
        return inputs, ouputs, context_prog

    @serving
    def classification(self,
                       images=None,
                       paths=None,
                       batch_size=1,
                       use_gpu=False):
        """
        API for image classification.

        Args:
            images (numpy.ndarray): data of images, shape of each is [H, W, C].
            paths (list[str]): The paths of images.
            batch_size (int): batch size.
            use_gpu (bool): Whether to use gpu.

        Returns:
            res (list[list[dict]]): The classfication results.
        """
        all_data = list()
        for yield_data in reader(images, paths):
            all_data.append(yield_data)

        total_num = len(all_data)
        loop_num = int(np.ceil(total_num / batch_size))

        res = list()
        for iter_id in range(loop_num):
            batch_data = list()
            handle_id = iter_id * batch_size
            for image_id in range(batch_size):
                try:
                    batch_data.append(all_data[handle_id + image_id])
                except:
                    pass
            # feed batch image
            batch_image = np.array([data['image'] for data in batch_data])
            batch_image = PaddleTensor(batch_image.copy())
            predictor_output = self.gpu_predictor.run([
                batch_image
            ]) if use_gpu else self.cpu_predictor.run([batch_image])
            out = postprocess(
                data_out=predictor_output[0].as_ndarray(),
                label_list=self.label_list)
            res.append(out)
        return res

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(
            description="Run the resnet50vd_animal module.",
            prog='hub run resnet50vd_animal',
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
        results = self.classification(
            paths=[args.input_path],
            batch_size=args.batch_size,
            use_gpu=args.use_gpu)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument(
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help="whether use GPU or not.")
        self.arg_config_group.add_argument(
            '--batch_size',
            type=ast.literal_eval,
            default=1,
            help="batch size.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument(
            '--input_path', type=str, help="path to image.")
