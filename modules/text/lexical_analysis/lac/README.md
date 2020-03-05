# LAC API 说明

## context(trainable=False)

获取lac的预训练program以及program的输入输出变量

**参数**

* trainable(bool): trainable=True表示program中的参数在Fine-tune时需要微调，否则保持不变

**返回**

* inputs(dict): program的输入变量
* outputs(dict): program的输出变量
* main_program(Program): 带有预训练参数的program

## set_user_dict(dict_path)

加载用户自定义词典

**参数**

* dict_path(str ): 自定义词典路径

## del_user_dict()

删除自定义词典

## lexical_analysis(texts=[], data={}, use_gpu=False, batch_size=1, user_dict=None, return_tag=True)

lac预测接口，预测输入句子的分词结果

**参数**

* texts(list): 待预测数据，如果使用texts参数，则不用传入data参数，二选一即可
* data(dict): 预测数据，key必须为text，value是带预测数据。如果使用data参数，则不用传入texts参数，二选一即可。建议使用texts参数，data参数后续会废弃。
* use_gpu(bool): 是否使用GPU预测
* batch_size(int): 批处理大小
* user_dict(None): 该参数不推荐使用，请在使用lexical_analysis()方法之前调用set_user_dict()方法设置自定义词典

**返回**

* results(dict): 分词结果

## get_tags()

获取lac的标签

**返回**

* tag_name_dict(dict): lac的标签
