# coding=utf-8
# @Author: 莫冉
# @Date: 2021-01-11
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import warnings
warnings.filterwarnings("ignore")

import json
from myCoTK.dataloader.bert_dataloader import GPTGen

# 只需要提供数据所在的路径即可，内部默认文件名是.json结尾的
# 默认包含train, dev, test这3个文件
file_path = "../data/film"
# 对应BERT词表的名称
bert_vocab_name = r"E:\Models\CDial-GPT2_LCCC-base\vocab.txt"
do_lower_case = True
num_turns = 5      # 包含当前轮，意思是使用之前的4轮
max_sent_length = 256
batch_size = 16



data_generator = GPTGen(file_id=file_path, vocab_name=bert_vocab_name,
                        do_lower_case=do_lower_case, max_sent_length=max_sent_length,
                        num_turns=num_turns)
# 初始化数据
data_generator.restart(key="test", batch_size=batch_size, shuffle=False)
# 获取一个batch查看
batch_data = data_generator.get_next_batch("test")

# 打印结果
for i in range(2):
    print(f"第{i}个样例@input_ids: ", data_generator.convert_bert_ids_to_tokens(batch_data["input_ids"][i], trim=False))
    # print(f"第{i}个样例@input_mask: ", batch_data["input_mask"][i])
    print(f"第{i}个样例@token_type_ids: ", batch_data["token_type_ids"][i])
    print(f"第{i}个样例@turn_ids: ", batch_data["turn_ids"][i])
    # print(f"第{i}个样例@lm_labels: ", batch_data["lm_labels"][i])
    print(f"第{i}个样例@posts_len: ", batch_data["posts_lens"][i])