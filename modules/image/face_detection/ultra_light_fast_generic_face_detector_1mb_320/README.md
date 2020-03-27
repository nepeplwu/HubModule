## 命令行预测

```
hub run ultra_light_fast_generic_face_detector_1mb_320 --input_path "/PATH/TO/IMAGE"
```

## API 说明

### face_detection

**API 定义**

```python
def face_detection(images=None,
                   paths=None,
                   batch_size=1,
                   use_gpu=False,
                   visualization=False,
                   output_dir=None,
                   confs_threshold=0.5,
                   iou_threshold=0.5):
```

**参数**

* images (list[numpy.ndarray]): 图片数据，ndarray.shape 为 [H, W, C]；
* paths (list[str]): 图片的路径；
* batch\_size (int): batch 的大小；
* use\_gpu (bool): 是否使用 GPU；
* visualization (bool): 是否将识别结果保存为图片文件；
* output\_dir (str): 图片的保存路径，当为 None 时，默认设为face\_detection\_result；
* confs\_threshold (float): 置信度的阈值；
* iou\_threshold (float): 人脸检测的置信度的阈值。

**返回**

* res (list[collections.OrderedDict]): 识别结果的列表，列表中每一个元素为 OrderedDict，关键字有 path, data，其中：
  * path 字段为原输入图片的路径
  * data 字段为检测结果，类型为list，list的每一个元素为dict，其中['left', 'right', 'top', 'bottom'] 为人脸识别框，'confidence' 为此识别框置信度。

## 预测代码示例

```python
import paddlehub as hub

face_detector = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320")

# 设待处理的图片数据 ndarray 保存在列表 ndarray_list
# 设待处理的图片的路径保存在列表 path_list
result = face_detector.face_detection(
    images=ndarray_list,
    paths=path_list,
    batch_size=1,
    use_gpu=True,
    visualization=True)

print(result)
```

## 模型相关信息

### 模型代码

https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

### Module贡献者

[Jason](https://github.com/jiangjiajun)、[Channingss](https://github.com/Channingss)

### 依赖

paddlepaddle >= 1.6.0
paddlehub >= 1.6.0
