# -*- coding:utf-8 -*-
import paddle.fluid as fluid


def lstm_net(data,
             dict_dim,
             emb_dim=128,
             hid_dim=128,
             hid_dim2=96,
             class_dim=2,
             emb_lr=30.0):
    """
    Lstm net
    """
    # embedding layer
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(
            learning_rate=emb_lr, name='@HUB_senta_lstm@embedding_0.w_0'))
    # Lstm layer
    fc0 = fluid.layers.fc(
        input=emb,
        size=hid_dim * 4,
        param_attr=fluid.ParamAttr(name="@HUB_senta_lstm@fc_0.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_lstm@fc_0.b_0"))
    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0,
        size=hid_dim * 4,
        is_reverse=False,
        param_attr=fluid.ParamAttr(name="@HUB_senta_lstm@lstm_0.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_lstm@lstm_0.b_0"))
    # max pooling layer
    lstm_max = fluid.layers.sequence_pool(input=lstm_h, pool_type='max')
    lstm_max_tanh = fluid.layers.tanh(lstm_max)
    # full connect layer
    fc1 = fluid.layers.fc(
        input=lstm_max_tanh,
        size=hid_dim2,
        act='tanh',
        param_attr=fluid.ParamAttr(name="@HUB_senta_lstm@fc_1.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_lstm@fc_1.b_0"))
    # softmax layer
    prediction = fluid.layers.fc(
        input=fc1,
        size=class_dim,
        act='softmax',
        param_attr=fluid.ParamAttr(name="@HUB_senta_lstm@fc_2.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_lstm@fc_2.b_0"))

    return prediction, fc1
