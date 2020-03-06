# SentaLSTM API说明

## sentiment_classify(texts=[], data={}, use_gpu=False, batch_size=1)

senta_lstm预测接口，预测输入句子的情感分类(二分类，积极/消极）

**参数**

* texts(list): 待预测数据，如果使用texts参数，则不用传入data参数，二选一即可
* data(dict): 预测数据，key必须为text，value是带预测数据。如果使用data参数，则不用传入texts参数，二选一即可。建议使用texts参数，data参数后续会废弃。
* use_gpu(bool): 是否使用GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置
* batch_size(int): 批处理大小

**返回**

* results(list): 情感分类结果

## context(trainable=False)

获取senta_lstm的预训练program以及program的输入输出变量

**参数**

* trainable(bool): trainable=True表示program中的参数在Fine-tune时需要微调，否则保持不变

**返回**

* inputs(dict): program的输入变量
* outputs(dict): program的输出变量
* main_program(Program): 带有预训练参数的program

## get_labels()

获取senta_lstm的类别

**返回**

* labels(dict): senta_lstm的类别(二分类，积极/消极)

## get_vocab_path()

获取预训练时使用的词汇表

**返回**

* vocab_path(str): 词汇表路径
