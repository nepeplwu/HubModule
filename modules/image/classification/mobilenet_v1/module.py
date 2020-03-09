import os
import numpy as np
import paddlehub as hub
import paddle.fluid as fluid
from mobilenet_v1.mobilenet_v1 import MobileNet
from paddlehub.module.module import moduleinfo
from mobilenet_v1.processor import load_label_info
from mobilenet_v1.data_feed import test_reader


@moduleinfo(
    name="mobilenet_v1",
    version="2.0.0",
    type="cv/object_detection",
    summary="for test",
    author="paddle",
    author_email="paddlepaddle@baidu.com")
class MobuleNet_V1(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(
            self.directory, "MobileNetV1_pretrained")
        self.label_names = load_label_info(
            os.path.join(self.directory, "label_file.txt"))
        self.infer_prog = None
        self.pred_out = None

    def context(self,
                input_image=None,
                trainable=True,
                pretrained=False,
                param_prefix='',
                get_prediction=False,
                yolo_v3=False):
        """Distill the Head Features, so as to perform transfer learning.

        :param input_image: image tensor.
        :type input_image: <class 'paddle.fluid.framework.Variable'>
        :param trainable: whether to set parameters trainable.
        :type trainable: bool
        :param pretrained: whether to load default pretrained model.
        :type pretrained: bool
        :param param_prefix: the prefix of parameters in yolo_head and backbone
        :type param_prefix: str
        :param get_prediction: whether to get prediction,
            if True, outputs is {'bbox_out': bbox_out},
            if False, outputs is {'head_features': head_features}.
        :type get_prediction: bool
        """
        context_prog = input_image.block.program if input_image else fluid.Program(
        )
        with fluid.program_guard(context_prog):
            #with fluid.unique_name.guard():
            # image
            image = input_image if input_image else fluid.data(
                name='image',
                shape=[-1, 3, 224, 224],
                dtype='float32',
                lod_level=0)
            backbone = MobileNet(
                norm_decay=0.,
                conv_group_scale=1,
                conv_learning_rate=0.1,
                extra_block_filters=[[256, 512], [128, 256], [128, 256],
                                     [64, 128]],
                with_extra_blocks=not get_prediction,
                yolo_v3=yolo_v3)
            out = backbone(image)

            inputs = {'image': image}
            if get_prediction:
                outputs = {'pred_out': out[-1]}
            else:
                outputs = {'body_feats': out}

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        #for param in context_prog.global_block().iter_parameters():
        #    param.trainable = trainable
        #startup_program = fluid.Program()
        with fluid.program_guard(context_prog):
            if pretrained:

                def _if_exist(var):
                    return os.path.exists(
                        os.path.join(self.default_pretrained_model_path,
                                     var.name))

                if not param_prefix:
                    fluid.io.load_vars(
                        exe,
                        self.default_pretrained_model_path,
                        main_program=context_prog,
                        predicate=_if_exist)

            return inputs, outputs, context_prog

    def classification(self,
                       paths=None,
                       images=None,
                       use_gpu=False,
                       batch_size=1,
                       output_dir=None,
                       score_thresh=0.5):

        if self.infer_prog is None:
            inputs, outputs, self.infer_prog = self.context(
                trainable=False, pretrained=True, get_prediction=True)
            self.infer_prog = self.infer_prog.clone(for_test=True)
            self.pred_out = outputs['pred_out']

        place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)
        all_images = []
        paths = paths if paths else []
        for yield_data in test_reader(paths, images):
            all_images.append(yield_data)

        images_num = len(all_images)
        loop_num = int(np.ceil(images_num / batch_size))
        class_maps = load_label_info('./label_file.txt')
        res_list = []
        TOPK = 1
        for iter_id in range(loop_num):
            batch_data = []
            handle_id = iter_id * batch_size
            for image_id in range(batch_size):
                try:
                    batch_data.append(all_images[handle_id + image_id])
                except:
                    pass
            feed = {'image': np.array(batch_data).astype('float32')}
            result = exe.run(
                self.infer_prog,
                feed=feed,
                fetch_list=[self.pred_out],
                return_numpy=True)
            for i, res in enumerate(result[0]):
                pred_label = np.argsort(res)[::-1][:TOPK]
                class_name = class_maps[int(pred_label)]
                res_list.append([pred_label, class_name])
        return res_list
