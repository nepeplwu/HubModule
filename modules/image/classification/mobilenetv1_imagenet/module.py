import os
import numpy as np
import paddlehub as hub
import paddle.fluid as fluid
from mobilenetv1_imagenet.mobilenet_v1 import MobileNet 
from paddlehub.module.module import moduleinfo
from mobilenetv1_imagenet.processor import load_label_info
from mobilenetv1_imagenet.data_feed import test_reader
@moduleinfo(
    name="mobilenetv1_imagenet",
    version="1.1.0",
    type="cv/classification",
    summary="MobileNet is a image classfication model trained with ImageNet-2012 dataset.",
    author="paddle",
    author_email="paddle-dev@baidu.com"
)
class MobuleNet_V1(hub.Module):
    def _initialize(self):
        self.default_pretrained_model_path = os.path.join(self.directory, "MobileNetV1_pretrained")
        self.label_names = load_label_info(os.path.join(self.directory, "label_file.txt"))
        self.infer_prog = None
        self.pred_out = None
 
    def context(self,
                input_image=None,
                trainable=True,
                pretrained=False, 
                param_prefix='',
                get_prediction=False,
                yolo_v3=False,
                top_k=1):
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
        :param yolo_v3: whether to output layers which yolo_v3 needs
        :type yolo_v3: bool
        """
        context_prog = input_image.block.program if input_image else fluid.Program()
        with fluid.program_guard(context_prog):
            image = input_image if input_image else fluid.data(
                name='image', shape=[-1, 3, 224, 224], dtype='float32', lod_level=0)
            backbone = MobileNet(
                norm_decay=0.,
                conv_group_scale=1,
                conv_learning_rate=0.1,
                extra_block_filters=[[256, 512], [128, 256], [128, 256], [64, 128]],
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
        with fluid.program_guard(context_prog):
            if pretrained:
                def _if_exist(var):
                    return os.path.exists(os.path.join(self.default_pretrained_model_path, var.name))
                if not param_prefix:
                    fluid.io.load_vars(exe, self.default_pretrained_model_path, main_program=context_prog, predicate=_if_exist)
            else:
                exe.run(fluid.default_startup_program())
            return inputs, outputs, context_prog

    def classification(
            self, 
            paths=None,
            images=None,
            use_gpu=False,
            batch_size=1,
            output_dir=None,
            score_thresh=0.5):
        """API of Classification.
        :param paths: the path of images.
        :type paths: list, each element is correspond to the path of an image.
        :param images: data of images, [N, H, W, C]
        :type images: numpy.ndarray
        :param use_gpu: whether to use gpu or not.
        :type use_gpu: bool
        :param batch_size: bathc size.
        :type batch_size: int
        :param output_dir: the directory to store the detection result.
        :type output_dir: str
        :param score_thresh: the threshold of detection confidence.
        :type score_thresh: float
        """
        if self.infer_prog is None:
            inputs, outputs, self.infer_prog = self.context(
                trainable=False,
                pretrained=True,
                get_prediction=True)
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
        res_list = []
        TOPK=1
        for iter_id in range(loop_num):
            batch_data = []
            handle_id = iter_id * batch_size
            for image_id in range(batch_size):
                try:
                    batch_data.append(all_images[handle_id + image_id])
                except:
                    pass
            feed = {
                'image': np.array(batch_data).astype('float32')}
            result = exe.run(
                self.infer_prog,
                feed=feed,
                fetch_list=[self.pred_out],
                return_numpy=True)       
            for i, res in enumerate(result[0]):
                pred_label = np.argsort(res)[::-1][:TOPK]
                class_name = self.label_names[int(pred_label)]
                res_list.append([pred_label, class_name])
        return res_list
