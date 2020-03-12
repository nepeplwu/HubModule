# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import unittest
import numpy as np
import paddle.fluid as fluid
import paddlehub as hub


class TestStyleProjection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """Prepare the environment once before execution of all tests.\n"""
        self.style_projection = hub.Module(name="style_projection_coco_wikiart")

    @classmethod
    def tearDownClass(self):
        """clean up the environment after the execution of all tests.\n"""
        self.style_projection = None

    def setUp(self):
        "Call setUp() to prepare environment\n"
        self.test_prog = fluid.Program()

    def tearDown(self):
        "Call tearDown to restore environment.\n"
        self.test_prog = None

    def test_encoder_context(self):
        self.style_projection.encoder_context(pretrained=True)

    def test_decoder_context(self):
        self.style_projection.decoder_context()

    def test_style_transfer(self):
        with fluid.program_guard(self.test_prog):
            content_dir = '../../image_dataset/style_tranfer/content/'
            style_dir = '../../image_dataset/style_tranfer/style/'
            content_paths = [
                os.path.join(content_dir, f) for f in os.listdir(content_dir)
            ]
            style_paths = [
                os.path.join(style_dir, f) for f in os.listdir(style_dir)
            ]
            for style_path in style_paths:
                t1 = time.time()
                self.style_projection.style_transfer(
                    content_paths=[content_paths[0]],
                    style_paths=[style_path],
                    alpha=0.8,
                    use_gpu=True)
                t2 = time.time()
                print('\nCost time: {}'.format(t2 - t1))


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(TestStyleProjection('test_encoder_context'))
    suite.addTest(TestStyleProjection('test_decoder_context'))
    suite.addTest(TestStyleProjection('test_style_transfer'))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
