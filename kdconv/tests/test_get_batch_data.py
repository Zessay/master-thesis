# coding=utf-8
# @Author: 莫冉
# @Date: 2020-12-28
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import warnings
warnings.filterwarnings("ignore")

import json
from myCoTK.dataloader.bert_dataloader import MyBERTRetrieval
from myCoTK.dataloader.single_turn_dialog import MyLM

# 只需要提供数据所在的路径即可，内部默认文件名是.json结尾的
# 默认包含train, dev, test这3个文件
file_path = "../data/film"
# 对应BERT词表的名称
bert_vocab_name = r"E:\Models\chinese_rbt3_pytorch\vocab.txt"
do_lower_case = True
num_turns = 5      # 包含当前轮，意思是使用之前的4轮
max_sent_length = 256
batch_size = 16

# 测试对数据的加载
with open(file_path + "/dev.json", "r", encoding="utf-8") as f:
    datas = json.load(f)
print("测试数据加载成功")



# data_generator = MyBERTRetrieval(file_id=file_path,
#                                  bert_vocab_name=bert_vocab_name,
#                                  do_lower_case=do_lower_case,
#                                  max_sent_length=max_sent_length,
#                                  num_turns=num_turns)

data_generator = MyLM(file_id=file_path,
                      max_sent_length=max_sent_length)
data_generator.restart(key="train", batch_size=batch_size, shuffle=True)
# 获取一个batch查看结果
batched_data = data_generator.get_next_batch("train")

for i in range(2):
    print(data_generator.convert_ids_to_tokens(batched_data['post_allvocabs'][i].tolist(), trim=False))
    print(data_generator.convert_ids_to_tokens(batched_data['resp_allvocabs'][i].tolist(), trim=False))