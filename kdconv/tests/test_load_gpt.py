# coding=utf-8
# @Author: 莫冉
# @Date: 2021-01-13
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import torch
from transformers import GPT2Config
from myModels.modeling_gpt2 import GPT2LMHeadModel

model_checkpoint = r"E:\Models\CDial-GPT2_LCCC-base\pytorch_model.bin"
config_file = r"E:\Models\CDial-GPT2_LCCC-base\config.json"


# config = GPT2Config.from_json_file(config_file)
# config.n_turns = 8
# model = GPT2LMHeadModel(config)

# print(model._modules.items())


print("加载模型")

state_dict = torch.load(model_checkpoint, map_location="cpu")

# print(dir(state_dict))
print(state_dict.keys())

# metadata = getattr(state_dict, "_metadata", None)
#
# missing_keys = []
# unexpected_keys = []
# error_msgs = []
#
# state_dict = state_dict.copy()
# if metadata is not None:
#     state_dict._metadata = metadata
#
# i = 0
#
# def load(module, prefix=""):
#     local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
#     global i
#     print(f"第{i}次打印local_metadata：")
#     print(local_metadata)
#     module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys,
#                                  unexpected_keys, error_msgs)
#     for name, child in module._modules.items():
#         if child is not None:
#             load(child, prefix + name + ".")
#
#     i += 1
#
# load(model, prefix="" if hasattr(model, "transformer") else "transformer.")
#
# print("missing_keys: ", missing_keys)
# print("unexpected keys: ", unexpected_keys)
# print("error msgs: ", error_msgs)

#
# print(metadata)

# print("state_dict type: ", type(state_dict))
# print("state_dict keys: ", state_dict.keys())
