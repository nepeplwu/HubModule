# -*- coding:utf-8 -*-
import paddle.fluid as fluid


def bilstm_net(data,
               dict_dim,
               emb_dim=128,
               hid_dim=128,
               hid_dim2=96,
               class_dim=2,
               emb_lr=30.0):
    """
    Bi-Lstm net
    """
    # embedding layer
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(
            name="@HUB_senta_bilstm@embedding_0.w_0", learning_rate=emb_lr))

    # bi-lstm layer
    fc0 = fluid.layers.fc(
        input=emb,
        size=hid_dim * 4,
        param_attr=fluid.ParamAttr(name="@HUB_senta_bilstm@fc_0.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_bilstm@fc_0.b_0"))
    rfc0 = fluid.layers.fc(
        input=emb,
        size=hid_dim * 4,
        param_attr=fluid.ParamAttr(name="@HUB_senta_bilstm@fc_1.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_bilstm@fc_1.b_0"))
    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0,
        size=hid_dim * 4,
        is_reverse=False,
        param_attr=fluid.ParamAttr(name="@HUB_senta_bilstm@lstm_0.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_bilstm@lstm_0.b_0"))
    rlstm_h, c = fluid.layers.dynamic_lstm(
        input=rfc0,
        size=hid_dim * 4,
        is_reverse=True,
        param_attr=fluid.ParamAttr(name="@HUB_senta_bilstm@lstm_1.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_bilstm@lstm_1.b_0"))

    # extract last layer
    lstm_last = fluid.layers.sequence_last_step(input=lstm_h)
    rlstm_last = fluid.layers.sequence_last_step(input=rlstm_h)
    lstm_last_tanh = fluid.layers.tanh(lstm_last)
    rlstm_last_tanh = fluid.layers.tanh(rlstm_last)

    # concat layer
    lstm_concat = fluid.layers.concat(input=[lstm_last, rlstm_last], axis=1)
    # full connect layer
    fc1 = fluid.layers.fc(
        input=lstm_concat,
        size=hid_dim2,
        act='tanh',
        param_attr=fluid.ParamAttr(name="@HUB_senta_bilstm@fc_2.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_bilstm@fc_2.b_0"))
    # softmax layer
    prediction = fluid.layers.fc(
        input=fc1,
        size=class_dim,
        act='softmax',
        param_attr=fluid.ParamAttr(name="@HUB_senta_bilstm@fc_3.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_bilstm@fc_3.b_0"))

    return prediction, fc1
