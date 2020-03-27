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
* output\_dir (str, optional): 输出目录，默认为 transfer\_output；
* visualization (bool, optioanl): 是否将结果保存为图片，默认为 True。

## API 说明

### (1) `style_transfer`

预测API，用于风格。

**API 定义**

```python
def style_transfer(self,
                   images=None,
                   paths=None,
                   alpha=1,
                   use_gpu=False,
                   visualization=False,
                   output_dir='transfer_result'):
```

**参数**

* images (list[dict]): ndarray 格式的图片数据。每一个元素都为一个 dict，有关键字 content, styles, weights(可选)，相应取值为：
    * content (numpy.ndarray): 待转换的图片，shape 为 [H, W, C]；
    * styles (list[numpy.ndarray]) : 作为底色的风格图片组成的列表，各个图片数组的shape 都是 [H, W, C]；
    * weights (list[float], optioal) : 各个 style 对应的权重。当不设置 weights 时，默认各个 style 有着相同的权重；
* paths (list[str]): 图片的路径。每一个元素都为一个 dict，有关键字 content, styles, weights(可选)，相应取值为：
    * content (str): 待转换的图片的路径；
    * styles (list[str]) : 作为底色的风格图片的路径；
    * weights (list[float], optioal) : 各个 style 对应的权重。当不设置 weights 时，默认各个 style 有着相同的权重；
* alpha (float) : 转换的强度，[0, 1] 之间，默认值为1；
* use\_gpu (bool): 是否使用 GPU；
* visualization (bool): 是否将结果保存为图片，默认为 False;
* output\_dir (str): 图片的保存路径，默认设为 transfer\_result 。

**返回**

* res (list[collections.OrderedDict]): 识别结果的列表，列表中每一个元素为 OrderedDict，关键字有 date, save_path，相应取值为：
  * save\_path (str): 保存图片的路径；
  * data (numpy.ndarray): 风格转换的图片数据。

## 预测代码示例

```python
import paddlehub as hub

style = hub.Module(name="tylepro_artistic)

content_path = 'content_img.jpg'
style_1_path = 'style_1_path.jpg'
style_2_path = 'style_2_path.jpg'
style_weights = [0.3, 0.7]

result = style_transfer(
    paths={
        'content': content_path,
        'style': [style_1_path, style_2_path],
        'weights': style_weights},
    alpha=0.8,
    use_gpu=True,
    visualization=True,
    output_dir='transfer_ouput')

print(result)
```

## 模型相关信息

### 模型论文

[Parameter-Free Style Projection for Arbitrary Style Transfer](https://arxiv.org/abs/2003.07694)

### 依赖

paddlepaddle >= 1.6.0

paddlehub >= 1.6.0

