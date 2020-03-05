# -*- coding:utf-8 -*-
import paddle.fluid as fluid


def cnn_net(data,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2,
            win_size=3):
    """
    Conv net
    """
    # embedding layer
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(name="@HUB_senta_cnn@embedding_0.w_0"))

    # convolution layer
    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=win_size,
        act="tanh",
        pool_type="max",
        param_attr=fluid.ParamAttr(name="@HUB_senta_cnn@sequence_conv_0.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_cnn@sequence_conv_0.b_0"))
    # full connect layer
    fc_1 = fluid.layers.fc(
        input=[conv_3],
        size=hid_dim2,
        param_attr=fluid.ParamAttr(name="@HUB_senta_cnn@fc_0.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_cnn@fc_0.b_0"))

    # softmax layer
    prediction = fluid.layers.fc(
        input=[fc_1],
        size=class_dim,
        act="softmax",
        param_attr=fluid.ParamAttr(name="@HUB_senta_cnn@fc_1.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_cnn@fc_1.b_0"))

    return prediction, fc_1
