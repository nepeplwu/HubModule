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
"""Finetuning on classification task """

import argparse
import ast
import time

import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--name", type=str, default="ernie_tiny", help="")
args = parser.parse_args()
# yapf: enable.


class ENDataset(hub.dataset.GLUE):
    def get_train_examples(self):
        return self.train_examples[:8000]

    def get_dev_examples(self):
        return self.dev_examples[:1000]

    def get_test_examples(self):
        return self.test_examples[:1000]


class CNDataset(hub.dataset.ChnSentiCorp):
    def get_train_examples(self):
        return self.train_examples[:8000]

    def get_dev_examples(self):
        return self.dev_examples[:1000]

    def get_test_examples(self):
        return self.test_examples[:1000]


if __name__ == '__main__':
    name = args.name
    try:
        if "L_3" in name or name == "bert_multi_uncased_L_12_H_768_A_12":
            version = "1.0.0"
        elif name == "ernie":
            version = "1.2.0"
        else:
            version = "1.1.0"
        module = hub.Module(
            name=name.replace("L_12_H_768_A_12", "L-12_H-768_A-12").replace(
                "L_24_H_1024_A_16", "L-24_H-1024_A-16").replace(
                    "L_3_H_768_A_12", "L-3_H-768_A-12").replace(
                        "L_3_H_1024_A_16", "L-3:q_H-1024_A-16"),
            version=version)
        inputs, outputs, program = module.context(
            trainable=True, max_seq_len=args.max_seq_len)

        if "chinese" in name or name in ["ernie", "ernie_tiny"]:
            dataset = CNDataset()
        else:
            dataset = ENDataset()

        metrics_choices = ["acc"]
        reader = hub.reader.ClassifyReader(
            dataset=dataset,
            vocab_path=module.get_vocab_path(),
            max_seq_len=args.max_seq_len,
            sp_model_path=module.get_spm_path(),
            word_dict_path=module.get_word_dict_path())

        # Construct transfer learning network
        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_output" for token-level output.
        pooled_output = outputs["pooled_output"]

        # Setup feed list for data feeder
        # Must feed all the tensor of module need
        feed_list = [
            inputs["input_ids"].name,
            inputs["position_ids"].name,
            inputs["segment_ids"].name,
            inputs["input_mask"].name,
        ]

        # Select finetune strategy, setup config and finetune
        strategy = hub.AdamWeightDecayStrategy(learning_rate=args.learning_rate)

        # Setup runing config for PaddleHub Finetune API
        if "L_12" in name or name in [
                "ernie", "ernie_tiny", "ernie_v2_eng_base"
        ] or "L_3" in name:
            batch_size = 16
        else:
            batch_size = 8

        config = hub.RunConfig(
            use_data_parallel=True,
            use_cuda=True,
            num_epoch=2,
            batch_size=batch_size,
            checkpoint_dir="ckpt_%s" % name,
            strategy=strategy,
            eval_interval=100)

        # Define a classfication finetune task by PaddleHub's API
        cls_task = hub.TextClassifierTask(
            data_reader=reader,
            feature=pooled_output,
            feed_list=feed_list,
            num_classes=dataset.num_labels,
            config=config,
            metrics_choices=metrics_choices)

        # Finetune and evaluate by PaddleHub's API
        # will finish training, evaluation, testing, save model automatically

        start = time.time()
        cls_task.finetune_and_eval()
        print("%s ******************    finetune time: %s" %
              (name, time.time() - start))

        # predict_data = [[example.text_a, example.text_b]
        #                 for example in dataset.get_dev_examples()]
        #
        # start = time.time()
        # cls_task.predict(
        #     data=predict_data, return_result=True, accelerate_mode=False)
        # old_time = time.time() - start
        # print("%s ******************    predict time: %s" % (name, old_time))
        #
        # start = time.time()
        # cls_task.predict(
        #     data=predict_data, return_result=True, accelerate_mode=True)
        # new_time = time.time() - start
        # print(
        #     "%s ******************    accelerate time: %s, accelerate rate: %s"
        #     % (name, new_time, (old_time - new_time) / old_time))
    except Exception as e:
        print(e)
