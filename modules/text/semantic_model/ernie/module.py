#coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from paddlehub import BERTModule
from paddlehub.module.module import moduleinfo

from ernie.model.ernie import ErnieModel, ErnieConfig


@moduleinfo(
    name="ernie",
    version="1.2.0",
    summary=
    "Baidu's ERNIE, Enhanced Representation through kNowledge IntEgration, max_seq_len=512 when predtrained",
    author="baidu-nlp",
    author_email="nlp@baidu.com",
    type="nlp/semantic_model",
)
class Ernie(BERTModule):
    def _initialize(self):
        ernie_config_path = os.path.join(self.directory, "assets",
                                         "ernie_config.json")
        self.ernie_config = ErnieConfig(ernie_config_path)
        self.MAX_SEQ_LEN = 512
        self.params_path = os.path.join(self.directory, "assets", "params")
        self.vocab_path = os.path.join(self.directory, "assets", "vocab.txt")\

    def net(self,
            input_ids,
            position_ids,
            segment_ids,
            input_mask,
            task_ids=None):
        if not task_ids:
            self.ernie_config._config_dict['use_task_id'] = False
        ernie = ErnieModel(
            src_ids=input_ids,
            position_ids=position_ids,
            sentence_ids=segment_ids,
            input_mask=input_mask,
            config=self.ernie_config,
            use_fp16=False)
        pooled_output = ernie.get_pooled_output()
        sequence_output = ernie.get_sequence_output()
        return pooled_output, sequence_output


if __name__ == '__main__':
    test_module = Ernie()
