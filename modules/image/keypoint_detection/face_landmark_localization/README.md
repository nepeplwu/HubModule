## 命令行预测

```
hub run face_landmark_localization --input_path "/PATH/TO/IMAGE"
```

## API 说明

### keypoint_detection

预测API，识别出人脸关键点。

**API 定义**


```python
def keypoint_detection(self,
                       images=None,
                       paths=None,
                       batch_size=1,
                       use_gpu=False,
                       output_dir=None,
                       visualization=False)
```

**参数**

* images (list[numpy.ndarray]): 图片数据，ndarray.shape 为 [H, W, C]；
* paths (list[str]): 图片的路径；
* batch\_size (int): batch 的大小；
* use\_gpu (bool): 是否使用 GPU；
* visualization (bool): 是否将识别结果保存为图片文件；
* output\_dir (str): 图片的保存路径，当为 None 时，默认设为keypoint\_detection\_result。

**返回**

* res (list[collections.OrderedDict]): 识别结果的列表，列表元素为 OrderedDict, 有以下两个字段：
    * im\_path : 输入图片的路径；
    * points: 识别关键点的坐标。

### `__init__`

**API 定义**

```
def __init__(self, face_detector_module=None)
```

**参数**

* face\_detector\_module (class): 人脸定位模型，默认为 ultra\_light\_fast\_generic\_face\_detector\_1mb\_640.

### set_face_detector_module

**API 定义**

```
def set_face_detector_module(self, face_detector_module=None)
```

**参数**

* face\_detector\_module (class): 人脸定位模型。


**API 定义**

```
def get_face_detector_module(self)
```

**返回**

* 当前模型使用的人脸定位模型。


## 预测代码示例

```python
import paddlehub as hub

face_landmark = hub.Module(name="face_landmark_localization")

# 设待处理的图片数据 ndarray 保存在列表 ndarray_list
# 设待处理的图片的路径保存在列表 path_list
result = face_landmark.keypoint_detection(
    images=ndarray_list,
    paths=path_list,
    batch_size=1,
    use_gpu=True,
    visualization=True)

print(result)
```

## 模型相关信息

### 模型代码

https://github.com/lsy17096535/face-landmark

### 依赖

paddlepaddle >= 1.6.0

paddlehub >= 1.6.0

