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

from pyramidbox_lite_server.data_feed import reader
from pyramidbox_lite_server.processor import postprocess, base64_to_cv2


@moduleinfo(
    name="pyramidbox_lite_server",
    type="CV/face_detection",
    author="baidu-vis",
    author_email="paddle-dev@baidu.com",
    summary="PyramidBox-Lite-Server is a high-performance face detection model.",
    version="1.2.0")
class PyramidBoxLiteServer(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(
            self.directory, "pyramidbox_lite_server_face_detection")
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

    def face_detection(self,
                       images=None,
                       paths=None,
                       data=None,
                       use_gpu=False,
                       output_dir='pyramidbox_server_face_detect_output',
                       visualization=False,
                       shrink=0.8,
                       confs_threshold=0.6):
        """
        API for face detection.

        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]
            paths (list[str]): The paths of images.
            use_gpu (bool): Whether to use gpu.
            output_dir (str): The path to store output images.
            visualization (bool): Whether to save image or not.
            shrink (float): parameter to control the resize scale in preprocess.
            confs_threshold (float): confidence threshold.

        Returns:
            res (list[dict]): The result of face detection and save path of images.
        """
        if use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
            except:
                raise RuntimeError(
                    "Attempt to use GPU for prediction, but environment variable CUDA_VISIBLE_DEVICES was not set correctly."
                )

        # compatibility with older versions
        if data:
            if 'image' in data:
                if paths is None:
                    paths = list()
                paths += data['image']
            elif 'data' in data:
                if images is None:
                    images = list()
                images += data['data']

        res = list()
        # process one by one
        for element in reader(images, paths, shrink):
            image = np.expand_dims(element['image'], axis=0).astype('float32')
            image_tensor = PaddleTensor(image.copy())
            data_out = self.gpu_predictor.run([
                image_tensor
            ]) if use_gpu else self.cpu_predictor.run([image_tensor])
            out = postprocess(
                data_out=data_out[0].as_ndarray(),
                org_im=element['org_im'],
                org_im_path=element['org_im_path'],
                image_width=element['image_width'],
                image_height=element['image_height'],
                output_dir=output_dir,
                visualization=visualization,
                shrink=shrink,
                confs_threshold=confs_threshold)
            res.append(out)
        return res

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        images_decode = [base64_to_cv2(image) for image in images]
        results = self.face_detection(images_decode, **kwargs)
        return results

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(
            description="Run the {} module.".format(self.name),
            prog='hub run {}'.format(self.name),
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
        results = self.face_detection(
            paths=[args.input_path],
            use_gpu=args.use_gpu,
            output_dir=args.output_dir,
            visualization=args.visualization,
            shrink=args.shrink,
            confs_threshold=args.confs_threshold)
        return results

    def add_module_config_arg(self):
        """
        Add the command config options.
        """
        self.arg_config_group.add_argument(
            '--use_gpu',
            type=ast.literal_eval,
            default=False,
            help="whether use GPU or not")
        self.arg_config_group.add_argument(
            '--output_dir',
            type=str,
            default='pyramidbox_server_face_detect_output',
            help="The directory to save output images.")
        self.arg_config_group.add_argument(
            '--visualization',
            type=ast.literal_eval,
            default=False,
            help="whether to save output as images.")

    def add_module_input_arg(self):
        """
        Add the command input options.
        """
        self.arg_input_group.add_argument(
            '--input_path', type=str, help="path to image.")
        self.arg_input_group.add_argument(
            '--shrink',
            type=ast.literal_eval,
            default=0.8,
            help=
            "resize the image to shrink * original_shape before feeding into network."
        )
        self.arg_input_group.add_argument(
            '--confs_threshold',
            type=ast.literal_eval,
            default=0.6,
            help="confidence threshold.")
