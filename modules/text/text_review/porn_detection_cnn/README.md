# PornDetectionCNN API说明

## detection(texts=[], data={}, use_gpu=False, batch_size=1)

porn_detection_cnn预测接口，鉴定输入句子是否为黄雯

**参数**

* texts(list): 待预测数据，如果使用texts参数，则不用传入data参数，二选一即可
* data(dict): 预测数据，key必须为text，value是带预测数据。如果使用data参数，则不用传入texts参数，二选一即可。建议使用texts参数，data参数后续会废弃。
* use_gpu(bool): 是否使用GPU预测
* batch_size(int): 批处理大小

**返回**

* results(list): 鉴定结果

## context(trainable=False)

获取porn_detection_cnn的预训练program以及program的输入输出变量

**参数**

* trainable(bool): trainable=True表示program中的参数在Fine-tune时需要微调，否则保持不变

**返回**

* inputs(dict): program的输入变量
* outputs(dict): program的输出变量
* main_program(Program): 带有预训练参数的program

## get_labels()

获取porn_detection_cnn的类别

**返回**

* labels(dict): porn_detection_cnn的类别
