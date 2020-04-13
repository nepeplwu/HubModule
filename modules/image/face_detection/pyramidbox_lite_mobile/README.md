```shell
$ hub install pyramidbox_lite_mobile==1.2.0
```

## 命令行预测

```
hub run pyramidbox_lite_mobile --input_path "/PATH/TO/IMAGE"
```

## API

```python
def face_detection(images=None,
                   paths=None,
                   data=None,
                   use_gpu=False,
                   output_dir='pyramidbox_mobile_face_detect_output',
                   visualization=False,
                   shrink=0.8,
                   confs_threshold=0.6)
```

检测输入图片中的所有人脸位置

**参数**

* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
* paths (list\[str\]): 图片的路径；
* batch\_size (int): batch 的大小；
* use\_gpu (bool): 是否使用 GPU；
* visualization (bool): 是否将识别结果保存为图片文件；
* output\_dir (str): 图片的保存路径，当为 None 时，默认设为face\_detector\_320\_predict\_output；
* shrink (float): 用于设置图片的缩放比例，该值越大，则对于输入图片中的小尺寸人脸有更好的检测效果（模型计算成本越高），反之则对于大尺寸人脸有更好的检测效果。
* confs\_threshold (float): 置信度的阈值。

**返回**

* res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，各字段为：
  * path (str): 原输入图片的路径；
  * data (list): 检测结果，list的每一个元素为 dict，各字段为:
      * confidence (float): 识别的置信度；
      * left (int): 边界框的左上角x坐标；
      * top (int): 边界框的左上角y坐标；
      * right (int): 边界框的右下角x坐标；
      * bottom (int): 边界框的右下角y坐标。


## 代码示例

```python
import paddlehub as hub
import cv2

face_detector = hub.Module(name="pyramidbox_lite_mobile")
result = face_detector.face_detection(images=[cv2.imread('/PATH/TO/IMAGE')])
# or
# result = face_detector.face_detection((paths=['/PATH/TO/IMAGE'])
```

## 服务部署

PaddleHub Serving可以部署一个在线人脸检测服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m pyramidbox_lite_mobile
```

这样就完成了一个人脸检测服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

## 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json
import cv2
import base64


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


# 发送HTTP请求
data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))]}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/pyramidbox_lite_mobile"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```

### 查看代码

[Paddle Models 人脸检测](https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/face_detection)

### 依赖

paddlepaddle >= 1.6.2

paddlehub >= 1.6.0
