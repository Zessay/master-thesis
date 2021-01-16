# coding=utf-8
# @Author: 莫冉
# @Date: 2021-01-15
from __future__ import print_function
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import json
import copy
import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, GPT2Config
from transformers import AdamW
from transformers.activations import ACT2FN
from transformers.file_utils import add_start_docstrings_to_model_forward, add_code_sample_docstrings
from transformers.modeling_outputs import CausalLMOutputWithPastAndCrossAttentions

from myModels.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel, GPT2_INPUTS_DOCSTRING
from myCoTK.bert_dataloader import GPTGenKA
from utils.cache_helper import try_cache
from utils.common import seed_everything, save_losses, masked_softmax, get_mask_from_sequence_lengths
from utils.MyMetrics import MyMetrics, MyPerplexity


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


_CONFIG_FOR_DOC = "GPT2Config"
_TOKENIZER_FOR_DOC = "BertTokenizer"

SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[speaker1]", "[speaker2]"]


def get_kg_mask(length_tensor: torch.Tensor, kg_len: int):
    """
    length_tensor: [batch, max_kg_num]，表示每一个kg的长度
    kg_len: int型，表示知识的最大长度
    """
    batch_size = length_tensor.shape[0]
    # [batch_size, max_kg_num, kg_length]
    length_mask_onehot = torch.nn.functional.one_hot(length_tensor, num_classes=kg_len + 1)[:, :, 1:].float()

    cumsum = torch.cumsum(length_mask_onehot, dim=2)
    length_ones = torch.ones_like(length_mask_onehot, dtype=torch.float32)
    origin_mask = length_ones - cumsum + length_mask_onehot
    return origin_mask.view(batch_size, -1, kg_len, 1)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def preprocess_batch(data, device=None):
    for key in ['input_ids', 'input_mask', 'token_type_ids', 'turn_ids', 'lm_labels',
                'kg_hr_length', 'kg_hrt_length', 'kg', 'kg_index', "posts_lens"]:
        if "labels" not in key and key != "kg_index":
            data[key] = torch.LongTensor(data[key])
        else:
            data[key] = torch.FloatTensor(data[key])

        if device is not None:
            data[key] = data[key].to(device)


class GPTGenKAModel(GPT2PreTrainedModel):
    def __init__(self, config, init_embeddings):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.init_wights()

        # 用于知识表征的词向量矩阵
        self.know_vocab_size, self.embed_size = np.shape(init_embeddings)
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(init_embeddings), freeze=False)

        # 用于最终分类层的分类器
        self.classifier = nn.Linear(config.n_embed + self.embed_size, config.vocab_size, bias=False)
        self.A = nn.Parameter(torch.Tensor(config.n_embed, self.embed_size))
        self.bias = nn.Parameter(torch.Tensor(1))

        # 对加权之后的知识向量进行一次gelu和layernorm
        self.know_activation = ACT2FN["gelu"]
        self.layer_norm = nn.LayerNorm(self.embed_size, eps=config.layer_norm_epsilon)

        nn.init.xavier_normal_(self.A)
        self.bias.data.fill_(0)


    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="gpt2",
        output_type=CausalLMOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(self,
                data,
                past_key_values=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                reduction="mean"):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=data["input_ids"],
            past_key_values=past_key_values,
            attention_mask=data["input_mask"],
            token_type_ids=data["token_type_ids"],
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            turn_ids=data["turn_ids"],
            source_ids=None
        )

        # 输出整个序列的表征，[batch, seq_len, n_embed]
        hidden_states = transformer_outputs[0]

        seq_len = hidden_states.size(1)
        # [batch, seq_len]
        context_mask = get_mask_from_sequence_lengths(data["posts_lens"], max_length=seq_len)
        context_embed = hidden_states * context_mask.unsqueeze(dim=-1)
        # [batch, n_embed]
        context_embed = torch.sum(context_embed, dim=1) / torch.max(
            data["posts_lens"].unsqueeze(-1), torch.ones_like(data["posts_lens"].unsqueeze(dim=-1), dtype=torch.float32))

        # --------------------------- 计算知识表征 -----------------------------
        batch_size, kg_len = data["kg"].size(0), data["kg"].size(2)

        # 输出知识对应的mask矩阵, [batch_size, max_kg_num, max_kg_hrt_length, 1]
        kg_hrt_mask = get_kg_mask(data["kg_hrt_length"], kg_len)

        # 将kg中的单词转化为embedding
        # [batch, max_kg_num, max_kg_hrt_length, embed_dim]
        kg_input = self.embed(data["kg"])

        # 计算三元组的均值向量，[batch, max_kg_num, embed_dim]
        kg_hrt_avg = torch.sum(kg_input * kg_hrt_mask, dim=2) / torch.max(
            torch.sum(kg_hrt_mask, dim=2), torch.ones_like(data["kg_hrt_length"].unsqueeze(-1), dtype=torch.float32))

        # ------------------------- 计算attention --------------------------
        # [batch, 1, embed_dim]
        intermediate = context_embed.mm(self.A).unsqueeze(dim=1)
        # [batch, max_kg_num]
        kg_alignment = torch.matmul(intermediate, kg_hrt_avg.transponse(1, 2)).squeeze(dim=1) + self.bias
        kg_mask = (data["kg_hrt_length"] > 0).to(torch.bool)   # [batch, max_kg_num]
        # [batch, max_kg_num]
        kg_alignment = masked_softmax(kg_alignment, kg_mask, dim=-1, memory_efficient=True)







































