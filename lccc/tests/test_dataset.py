# coding=utf-8
# @Author: 莫冉
# @Date: 2020-12-02
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import logging
from pathlib import Path
from transformers import BertTokenizer

# 导入数据加载的包
from od.inputters.inputter import get_data
from od.inputters.dataset_wb import WBDataset


# 定义日志
logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

model_path = r"E:\Models\CDial-GPT_LCCC-base"
vocab_file = "vocab.txt"

dataset_path = "../data/toy_data.json"
dataset_cache = "../data/dataset_cache"

tokenizer = BertTokenizer.from_pretrained(Path(model_path) / vocab_file)

# 获取将单词转化为索引之后的数据集
datasets, raw_samples = get_data(tokenizer, dataset_path, dataset_cache, logger)

print("转化为id之后的输入数据...")
print(datasets["train"][0])

# 对数据进行分装
train_dataset = WBDataset(datasets["train"], tokenizer)

print("获取每一个单词对应的标签")
print("input_ids: ", train_dataset[0]["input_ids"])
print("lm_labels: ", train_dataset[0]["lm_labels"])