## 命令行预测

```
hub run stylepro_artistic --选项 选项值
```

**选项说明：**

* content (str): 待转换风格的图片的存放路径；
* styles (str): 作为底色的风格图片的存放路径，不同图片用英文逗号 `,` 间隔；
* weights (float, optional) : styles 的权重，用英文逗号 `,` 间隔；
* alpha (float, optioal)：转换的强度，[0, 1] 之间，默认值为1；
* use\_gpu (bool, optional): 是否使用 gpu，默认为 False；
* output\_dir (str, optional): 输出目录，默认为 transfer\_result；
* visualization (bool, optioanl): 是否将结果保存为图片，默认为 True。

## API

```python
def style_transfer(self,
                   images=None,
                   paths=None,
                   alpha=1,
                   use_gpu=False,
                   visualization=False,
                   output_dir='transfer_result'):
```

对图片进行风格转换

**参数**

* images (list[dict]): ndarray 格式的图片数据。每一个元素都为一个 dict，有关键字 content, styles, weights(可选)，相应取值为：
    * content (numpy.ndarray): 待转换的图片，shape 为 [H, W, C]，BGR格式；
    * styles (list[numpy.ndarray]) : 作为底色的风格图片组成的列表，各个图片数组的shape 都是 [H, W, C]，BGR格式；
    * weights (list[float], optioal) : 各个 style 对应的权重。当不设置 weights 时，默认各个 style 有着相同的权重；
* paths (list[str]): 图片的路径。每一个元素都为一个 dict，有关键字 content, styles, weights(可选)，相应取值为：
    * content (str): 待转换的图片的路径；
    * styles (list[str]) : 作为底色的风格图片的路径；
    * weights (list[float], optioal) : 各个 style 对应的权重。当不设置 weights 时，各个 style 的权重相同；
* alpha (float) : 转换的强度，[0, 1] 之间，默认值为1；
* use\_gpu (bool): 是否使用 GPU；
* visualization (bool): 是否将结果保存为图片，默认为 False;
* output\_dir (str): 图片的保存路径，默认设为 transfer\_result 。

**返回**

* res (list[dict]): 识别结果的列表，列表中每一个元素为 OrderedDict，关键字有 date, save_path，相应取值为：
  * save\_path (str): 保存图片的路径；
  * data (numpy.ndarray): 风格转换的图片数据。

## 代码示例

```python
import paddlehub as hub
import cv2

stylepro_artistic = hub.Module(name="stylepro_artistic")
result = stylepro_artistic.style_transfer(
    images=[{
        'content': cv2.imread('/PATH/TO/CONTENT_IMAGE'),
        'styles': [cv2.imread('/PATH/TO/STYLE_IMAGE')]
    }])

# or
# result = stylepro_artistic.style_transfer(
#     paths=[{
#         'content': '/PATH/TO/CONTENT_IMAGE',
#         'styles': ['/PATH/TO/STYLE_IMAGE']
#     }])
```

## 服务部署

PaddleHub Serving可以部署一个在线风格转换服务。

## 第一步：启动PaddleHub Serving

运行启动命令：
```shell
$ hub serving start -m stylepro_artistic
```

这样就完成了一个风格转换服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测，则需要在启动服务之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置。

## 第二步：发送预测请求

配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果

```python
import requests
import json
import cv2
import base64
import paddlehub as hub
import numpy as np

def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')

def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data

data = {'images':[
    {
        'content':cv2_to_base64(cv2.imread('/PATH/TO/CONTENT_IMAGE')),
        'styles':[cv2_to_base64(cv2.imread('/PATH/TO/STYLE_IMAGE'))]
    }
]}

headers = {"Content-type": "application/json"}
url = "http://127.0.0.1:8866/predict/stylepro_artistic"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

print(base64_to_cv2(r.json()["results"][0]['data']))
```

### 依赖

paddlepaddle >= 1.6.2

paddlehub >= 1.6.0