# SimnetBOW API说明

## similarity(texts=[], data={}, use_gpu=False, batch_size=1)

simnet_bow预测接口，计算两个句子的cosin相似度

**参数**

* texts(list): 待预测数据，第一个元素(list)为第一顺序句子，第二个元素(list)为第二顺序句子，两个元素长度相同。
如texts=[["这道题太难了", "这道题太难了", "这道题太难了"], ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]]。
如果使用texts参数，则不用传入data参数，二选一即可
* data(dict): 预测数据，key必须为'text_1' 和'text_2'，相应的value(list)是第一顺序句子和第二顺序句子。
如data={"text_1": ["这道题太难了", "这道题太难了", "这道题太难了"], "text_2": ["这道题是上一年的考题", "这道题不简单", "这道题很有意思"]}。
如果使用data参数，则不用传入texts参数，二选一即可。建议使用texts参数，data参数后续会废弃。
* data(dict): 预测数据，key必须为'text_1' 和'text_2'，相应的value(list)是第一顺序句子和第二顺序句子。如果使用data参数，则不用传入texts参数，二选一即可。建议使用texts参数，data参数后续会废弃。
* use_gpu(bool): 是否使用GPU预测，如果使用GPU预测，则在预测之前，请设置CUDA_VISIBLE_DEVICES环境变量，否则不用设置
* batch_size(int): 批处理大小

**返回**

* results(list): 带预测数据的cosin相似度

## context(trainable=False)

获取simnet_bow的预训练program以及program的输入输出变量

**参数**

* trainable(bool): trainable=True表示program中的参数在Fine-tune时需要微调，否则保持不变

**返回**

* inputs(dict): program的输入变量
* outputs(dict): program的输出变量
* program(Program): 带有预训练参数的program

## get_vocab_path()

获取预训练时使用的词汇表

**返回**

* vocab_path(str): 词汇表路径
