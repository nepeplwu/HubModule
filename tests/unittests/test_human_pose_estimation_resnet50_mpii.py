# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import unittest

import cv2
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub

pic_dir = '../image_dataset/keypoint_detection/'


class TestHumanPoseEstimation(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests.\n"""
        self.pose = hub.Module(name="human_pose_estimation_resnet50_mpii")

    @classmethod
    def tearDownClass(self):
        """clean up the environment after the execution of all tests.\n"""
        self.pose = None

    def setUp(self):
        "Call setUp() to prepare environment\n"
        self.test_prog = fluid.Program()

    def tearDown(self):
        "Call tearDown to restore environment.\n"
        self.test_prog = None

    def test_single_pic(self):
        with fluid.program_guard(self.test_prog):
            pics_path_list = [
                os.path.join(pic_dir, f) for f in os.listdir(pic_dir)
            ]
            t1 = time.time()
            for pic_path in pics_path_list:
                result = self.pose.keypoint_detection(
                    paths=[pic_path], use_gpu=True, visualization=True)
            t2 = time.time()
            print('It cost {} seconds when batch_size=1.'.format(t2 - t1))

    def test_batch(self):
        with fluid.program_guard(self.test_prog):
            pics_path_list = [
                os.path.join(pic_dir, f) for f in os.listdir(pic_dir)
            ]
            t1 = time.time()
            result = self.pose.keypoint_detection(
                paths=pics_path_list,
                batch_size=3,
                output_dir='batch_output',
                use_gpu=True,
                visualization=True)
            t2 = time.time()
            print('It cost {} seconds when batch_size=3.'.format(t2 - t1))
            print(result)

    def test_ndarray(self):
        with fluid.program_guard(self.test_prog):
            pics_path_list = [
                os.path.join(pic_dir, f) for f in os.listdir(pic_dir)
            ]
            pics_ndarray = list()
            for pic_path in pics_path_list:
                im = cv2.imread(pic_path)
                result = self.pose.keypoint_detection(
                    images=np.expand_dims(im, axis=0),
                    output_dir='ndarray_output',
                    use_gpu=True,
                    visualization=True)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestHumanPoseEstimation('test_single_pic'))
    suite.addTest(TestHumanPoseEstimation('test_batch'))
    suite.addTest(TestHumanPoseEstimation('test_ndarray'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
