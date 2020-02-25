# -*- coding:utf-8 -*-
import paddle.fluid as fluid


def bilstm_net(data,
               label,
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
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, acc, prediction, fc1


def bow_net(data,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2):
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
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, acc, prediction, fc_2


def cnn_net(data,
            label,
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
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, acc, prediction, fc_1


def lstm_net(data,
             label,
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
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, acc, prediction, fc1


def gru_net(data,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2,
            emb_lr=30.0):
    """
    gru net
    """
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(
            learning_rate=emb_lr, name='@HUB_senta_gru@embedding_0.w_0'))
    fc0 = fluid.layers.fc(
        input=emb,
        size=hid_dim * 3,
        param_attr=fluid.ParamAttr(name="@HUB_senta_gru@fc_0.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_gru@fc_0.b_0"))
    gru_h = fluid.layers.dynamic_gru(
        input=fc0,
        size=hid_dim,
        is_reverse=False,
        param_attr=fluid.ParamAttr(name="@HUB_senta_gru@gru_0.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_gru@gru_0.b_0"))
    gru_max = fluid.layers.sequence_pool(input=gru_h, pool_type='max')
    gru_max_tanh = fluid.layers.tanh(gru_max)
    fc1 = fluid.layers.fc(
        input=gru_max_tanh,
        size=hid_dim2,
        act='tanh',
        param_attr=fluid.ParamAttr(name="@HUB_senta_gru@fc_1.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_gru@fc_1.b_0"))
    prediction = fluid.layers.fc(
        input=fc1,
        size=class_dim,
        act='softmax',
        param_attr=fluid.ParamAttr(name="@HUB_senta_gru@fc_2.w_0"),
        bias_attr=fluid.ParamAttr(name="@HUB_senta_gru@fc_2.b_0"))
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, acc, prediction, fc1
