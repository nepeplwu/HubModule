# coding:utf-8
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddlehub as hub

module = hub.Module(name="ernie")
res = module.get_embedding(
    texts=[['飞桨（PaddlePaddle）是国内开源产业级深度学习平台', 'PaddleHub是飞桨生态的预训练模型应用工具'],
           ["飞浆PaddleHub"]],
    batch_size=10)
print(res[0][0].shape, res[0][1].shape)

print(len(res))

print(list(set(module.get_params_layer().values())))
