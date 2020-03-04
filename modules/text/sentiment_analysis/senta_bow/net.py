# -*- coding:utf-8 -*-
import paddle.fluid as fluid


def bow_net(data, dict_dim, emb_dim=128, hid_dim=128, hid_dim2=96, class_dim=2):
    """
    Bow net
    """
    # embedding layer
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(name="@HUB_senta_bow@embedding_0.w_0"))

    # bow layer
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow)
    # full connect layer
    fc_1 = fluid.layers.fc(
        input=bow_tanh,
        size=hid_dim,
        act="tanh",
        param_attr=fluid.ParamAttr(name="@HUB_senta_bow@fc_0.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_bow@fc_0.b_0"))
    fc_2 = fluid.layers.fc(
        input=fc_1,
        size=hid_dim2,
        act="tanh",
        param_attr=fluid.ParamAttr(name="@HUB_senta_bow@fc_1.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_bow@fc_1.b_0"))

    # softmax layer
    prediction = fluid.layers.fc(
        input=[fc_2],
        size=class_dim,
        act="softmax",
        param_attr=fluid.ParamAttr(name="@HUB_senta_bow@fc_2.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_bow@fc_2.b_0"))

    return prediction, fc_2
