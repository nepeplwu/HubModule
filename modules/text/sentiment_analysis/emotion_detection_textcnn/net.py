# -*- coding:utf-8 -*-
import paddle.fluid as fluid


def textcnn_net(data,
                label,
                dict_dim,
                emb_dim=128,
                hid_dim=128,
                hid_dim2=96,
                class_dim=3,
                win_sizes=None):
    """
    Textcnn_net
    """
    if win_sizes is None:
        win_sizes = [1, 2, 3]

    # embedding layer
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(
            name="@HUB_emotion_detection_textcnn@embedding_0.w_0", ))

    # convolution layer
    convs = []
    for index, win_size in enumerate(win_sizes):
        conv_h = fluid.nets.sequence_conv_pool(
            input=emb,
            num_filters=hid_dim,
            filter_size=win_size,
            act="tanh",
            pool_type="max",
            param_attr=fluid.ParamAttr(
                name="@HUB_emotion_detection_textcnn@sequence_conv_%d.w_0" %
                (index)),
            bias_attr=fluid.ParamAttr(
                name="@HUB_emotion_detection_textcnn@sequence_conv_%d.b_0" %
                (index)))
        convs.append(conv_h)
    convs_out = fluid.layers.concat(input=convs, axis=1)

    # full connect layer
    fc_1 = fluid.layers.fc(
        input=[convs_out],
        size=hid_dim2,
        act="tanh",
        param_attr=fluid.ParamAttr(
            name="@HUB_emotion_detection_textcnn@fc_0.w_0"),
        bias_attr=fluid.ParamAttr(
            name="@HUB_emotion_detection_textcnn@fc_0.b_0"))
    # softmax layer
    prediction = fluid.layers.fc(
        input=[fc_1],
        size=class_dim,
        act="softmax",
        param_attr=fluid.ParamAttr(
            name="@HUB_emotion_detection_textcnn@fc_1.w_0"),
        bias_attr=fluid.ParamAttr(
            name="@HUB_emotion_detection_textcnn@fc_1.b_0"))
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, acc, prediction, fc_1
