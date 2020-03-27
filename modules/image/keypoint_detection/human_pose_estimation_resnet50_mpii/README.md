## 命令行预测

```
hub run human_pose_estimation_resnet50_mpii --input_path "/PATH/TO/IMAGE"
```

## API 说明

### (1) `keypoint_detection`

预测API，识别出人体骨骼关键点。

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

* res (list[collections.OrderedDict]): 识别结果的列表，列表元素为 OrderedDict, 为各个关节点的坐标。

## 预测代码示例

```python
import paddlehub as hub

pose = hub.Module(name="human_pose_estimation_resnet50_mpii")

# 设待处理的图片数据 ndarray 保存在列表 ndarray_list
# 设待处理的图片的路径保存在列表 path_list
result = pose.keypoint_detection(
    images=ndarray_list,
    paths=path_list,
    batch_size=1,
    use_gpu=True,
    visualization=True)

print(result)
```

## 模型相关信息

### 模型代码

https://github.com/PaddlePaddle/models/tree/develop/PaddleCV/human_pose_estimation

### 依赖

paddlepaddle >= 1.6.0

paddlehub >= 1.6.0

