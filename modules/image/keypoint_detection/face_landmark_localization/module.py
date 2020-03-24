# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import ast
import argparse
import time
import os
from collections import OrderedDict

import cv2
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub
from paddle.fluid.core import PaddleTensor, AnalysisConfig, create_paddle_predictor
from paddlehub.module.module import moduleinfo, runnable, serving

from face_landmark_localization.processor import check_dir, postprocess
from face_landmark_localization.data_feed import reader


@moduleinfo(
    name="face_landmark_localization",
    type="CV/keypoint_detection",
    author="paddlepaddle",
    author_email="paddle-dev@baidu.com",
    summary=
    "Face_Landmark_Localization can be used to locate face landmark. This Module is trained through the MPII Human Pose dataset.",
    version="1.1.0")
class FaceLandmarkLocalization(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(
            self.directory, "face_landmark_localization")
        self.face_detector = hub.Module(
            name="ultra_light_fast_generic_face_detector_1mb_640")
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

    @serving
    def keypoint_detection(self,
                           images=None,
                           paths=None,
                           use_gpu=False,
                           output_dir=None,
                           visualization=False,
                           face_detector_module=None):
        """
        API for human pose estimation and tracking.

        Args:
            images (numpy.ndarray): data of images, with shape [N, H, W, C].
            paths (list[str]): The paths of images.
            use_gpu (bool): Whether to use gpu.
            output_dir (str): The path to store output images.
            visualization (bool): Whether to save image or not.
            face_detector_module (class): module to detect face.

        Returns:
            res (list[collections.OrderedDict]): The key points of human pose.
        """
        # create output directory
        output_dir = output_dir if output_dir else os.path.join(
            os.getcwd(), 'keypoint_detection_result')
        check_dir(output_dir)

        res = list()
        face_detector_module = face_detector_module if face_detector_module else self.face_detector
        for element in reader(face_detector_module, images, paths, use_gpu):
            each_one = OrderedDict()
            each_one['im_path'] = element['org_im_path']
            im_save_path = os.path.join(output_dir, element['org_im_path'])
            cv2.imwrite(im_save_path, element['org_im'])
            each_one['points'] = list()
            for data_dict in element['image']:
                # postprocess face one by one
                face = np.expand_dims(data_dict['face'], axis=0)
                face_tensor = PaddleTensor(face.copy())
                pred_out = self.gpu_predictor.run([
                    face_tensor
                ]) if use_gpu else self.cpu_predictor.run([face_tensor])
                points = pred_out[0].as_ndarray().flatten()
                points_fixed = postprocess(
                    points=points,
                    face_x1=data_dict['x1'],
                    face_y1=data_dict['y1'],
                    face_x2=data_dict['x2'],
                    face_y2=data_dict['y2'],
                    im_path=im_save_path,
                    visualization=visualization)
                each_one['points'].append(points_fixed)
            res.append(each_one)
        return res

    @runnable
    def run_cmd(self, argvs):
        """
        Run as a command.
        """
        self.parser = argparse.ArgumentParser(
            description="Run the human_pose_estimation_resnet50_mpii module.",
            prog='hub run human_pose_estimation_resnet50_mpii',
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
        results = self.keypoint_detection(
            paths=[args.input_path],
            use_gpu=args.use_gpu,
            output_dir=args.output_dir,
            visualization=args.visualization)
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
            default=None,
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
