```shell
$ hub install ace2p==1.1.0
```

<p align="center">
<img src="https://bj.bcebos.com/paddlehub/paddlehub-img/ace2p_network.jpg" hspace='10'/> <br />
</p>

## 命令行预测

```
hub run ace2p --input_path "/PATH/TO/IMAGE"
```

## API

```python
def segmentation(images=None,
                 paths=None,
                 scale=(473, 473),
                 rotation=0,
                 batch_size=1,
                 use_gpu=False,
                 output_dir='ace2p_output',
                 visualization=False):
```

预测API，用于图像分割得到人体解析。

**参数**

* images (list\[numpy.ndarray\]): 图片数据，ndarray.shape 为 \[H, W, C\]，BGR格式；
* paths (list\[str\]): 图片的路径；
* scale (tuple): 预处理得到的图片的shape 为 [20, sacle[0], scale[1]]；
* rotation (int): 旋转角度，用于预处理中求解仿射矩阵；
* batch\_size (int): batch 的大小；
* use\_gpu (bool): 是否使用 GPU；
* visualization (bool): 是否将识别结果保存为图片文件；

**返回**

* res (list\[dict\]): 识别结果的列表，列表中每一个元素为 dict，关键字有 path, data，其中：
  * path 字段为原输入图片的路径（仅当使用paths输入时存在）；
  * data 字段为检测结果，类型为list，list的每一个元素为dict，其中'left', 'right', 'top', 'bottom' 为人脸识别框，'confidence' 为此识别框置信度。


## 代码示例

```python
import paddlehub as hub
import cv2

human_parser = hub.Module(name="ace2p")
result = human_parser.segmentation(images=[cv2.imread('/PATH/TO/IMAGE')])
# or
# result = human_parser.segmentation((paths=['/PATH/TO/IMAGE'])
```

## 服务部署

PaddleHub Serving可以部署一个人体解析的在线服务。

### 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m ace2p
```

这样就完成了一个人体解析服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA\_VISIBLE\_DEVICES环境变量，否则不用设置。

### 第二步：发送预测请求

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
url = "http://127.0.0.1:8866/predict/ace2p"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```

## 调色板

<p align="left">
<img src="https://bj.bcebos.com/paddlehub/paddlehub-img/ace2p_palette.jpg" hspace='10'/> <br />
</p>

## 依赖

paddlepaddle >= 1.6.2

paddlehub >= 1.6.0
