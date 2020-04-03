## 命令行预测

```
hub run deeplabv3p_xception65_humanseg --input_path "/PATH/TO/IMAGE"
```

## API

```python
def segmentation(self,
                 images=None,
                 paths=None,
                 batch_size=1,
                 use_gpu=False,
                 visualization=False,
                 output_dir=None)
```

预测API，用于人像分割。

**参数**

* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，GBR格式；
* paths (list\[str\]): 图片的路径；
* batch\_size (int): batch 的大小；
* use\_gpu (bool): 是否使用 GPU；
* visualization (bool): 是否将识别结果保存为图片文件；
* output\_dir (str): 图片的保存路径，当为 None 时，默认设为 segmentation\_result；

**返回**

* res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，关键字有 save\_path, data，对应的取值为：
  * save\_path (str, optional): 可视化图片的保存路径（仅当visualization=True时存在）；
  * data (numpy.ndarray): 人像分割处理后得到的图片数据。

## 代码示例

```python
import paddlehub as hub
import cv2

human_seg = hub.Module(name="deeplabv3p_xception65_humanseg")
result = human_seg.segmentation(images=[cv2.imread('/PATH/TO/IMAGE')])
# or
# result = human_seg.segmentation(paths=['/PATH/TO/IMAGE'])
```

## 服务部署

PaddleHub Serving可以部署一个在线人脸检测服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m deeplabv3p_xception65_humanseg
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
import paddlehub as hub

def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')

# 发送HTTP请求
data = {'images':[cv2_to_base64(cv2.imread("/PATH/TO/IMAGE"))]}
headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/deeplabv3p_xception65_humanseg"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```

### 查看代码

[PaddleSeg 特色垂类模型 - 人像分割](https://github.com/PaddlePaddle/PaddleSeg/tree/release/v0.4.0/contrib)

### 依赖

paddlepaddle >= 1.6.2

paddlehub >= 1.6.0
