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

from roberta_wwm_ext_chinese_base_distillation.model.bert import BertConfig, BertModel


@moduleinfo(
    name="roberta_wwm_ext_chinese_base_distillation",
    version="1.1.0",
    summary=
    "roberta_wwm_ext_chinese_base_distillation, 3-layer, 768-hidden, 12-heads, 38M parameters ",
    author="kinghuin",
    author_email="kinghuin_chull@163.com",
    type="nlp/semantic_model",
)
class BertWwm(BERTModule):
    def _initialize(self):
        self.MAX_SEQ_LEN = 512
        self.params_path = os.path.join(self.directory, "assets", "params")
        self.vocab_path = os.path.join(self.directory, "assets", "vocab.txt")

        bert_config_path = os.path.join(self.directory, "assets",
                                        "bert_config_rbt3.json")
        self.bert_config = BertConfig(bert_config_path)

    def net(self, input_ids, position_ids, segment_ids, input_mask):
        bert = BertModel(
            src_ids=input_ids,
            position_ids=position_ids,
            sentence_ids=segment_ids,
            input_mask=input_mask,
            config=self.bert_config,
            use_fp16=False)
        pooled_output = bert.get_pooled_output()
        sequence_output = bert.get_sequence_output()
        return pooled_output, sequence_output


if __name__ == '__main__':
    test_module = BertWwm()
