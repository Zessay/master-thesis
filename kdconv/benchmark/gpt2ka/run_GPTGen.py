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

from myModels.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel, GPT2_INPUTS_DOCSTRING
from myCoTK.bert_dataloader import GPTGenKA
from myCoTK.wordvector import TencentChinese
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

    def forward(self,
                data,
                labels=None,
                kg_alignment=None,
                knowledge_embed=None,
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

        if (kg_alignment is None) or (knowledge_embed is None):
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

            # ------------------------- 知识表征 --------------------------
            # 对知识表征加权, [batch, embed_dim]
            knowledge_embed = kg_alignment.unsqueeze(dim=1).bmm(kg_hrt_avg).squeeze(dim=1)

        # 找到attention分数最大的位置
        kg_max = torch.argmax(kg_alignment, dim=-1)   # [batch, ]，每一个样例关注的最多的知识
        kg_max_onehot = F.one_hot(kg_max, num_classes=kg_alignment.shape[1]).float()  # 将关注最多的知识对应位置设为1

        # -------------------- 将知识和输出表征拼接 ---------------------
        # GPT2的最后一层是先使用gleu激活，然后再layer_norm，这里和GPT2保持一致
        act_know = self.know_activation(knowledge_embed).unsqueeze(dim=1).expand(-1, seq_len, -1).contiguous()
        act_know = self.layer_norm(act_know)
        # 将两个输出拼接并经过分类层
        logits = self.classifier(torch.cat([hidden_states, act_know], dim=-1))

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:, :].contiguous()
            # 将按样本的数据展开成按照token的数据
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction=reduction)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # 计算kg关注度的损失
            kg_acc_num = torch.sum(torch.sum(data["kg_index"] * kg_max_onehot, dim=1))
            kg_all_num = torch.sum(torch.max(data["kg_index"], dim=1)[0])
            kg_acc = kg_acc_num / torch.max(kg_all_num, torch.FloatTensor([1]).to(device=kg_all_num.device))

            # 计算关注正确的kg的损失
            kg_loss = torch.mean(torch.sum(-torch.log(
                torch.clamp(kg_alignment, 1e-12, 1.0)) * data["kg_index"], dim=1) / torch.max(
                torch.sum(data["kg_index"], dim=1), torch.ones([data["kg_index"].size(0)], dtype=torch.float32).to(device=data["kg_index"].device)))

            return loss, kg_loss, kg_acc
        else:
            # 其中logits为[batch, vocab_size]
            # kg_alignment为[batch, max_kg_num]
            # knowledge_embed为[batch, embed_dim]
            return logits, kg_alignment, knowledge_embed




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt_config_file", default="CDial-GPT2_LCCC-base/config.json",
                        type=str, help="The config json file corresponding to the pre-trained GPT model.")
    parser.add_argument("--vocab_file", default="CDial-GPT2_LCCC-base/vocab.txt",
                        type=str, help="The vocabulary file that the GPT model was trained on.")
    parser.add_argument("--init_checkpoint", default="CDial-GPT2_LCCC-base/pytorch_model.bin",
                        type=str, help="Initial checkpoint (usually from a pre-trained GPT model).")

    # Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the predictions will be written.")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--cached_dir", default=None, type=str, required=True,
                        help="The output directory where the data cache will be written.")

    # Other parameters
    parser.add_argument("--name", type=str, default="GPT2LMHeadModel", help="name of the model")
    parser.add_argument("--dataset", type=str, default="GPTGen",
                        help="Dataloader class.")
    parser.add_argument("--datapath", type=str, default="resources://OpenSubtitles",
                        help="Directory for data set.")
    parser.add_argument("--wv_class", type=str, default="TencentChinese",
                        help="Wordvector class, none for not using pretrained wordvec.")
    parser.add_argument("--wv_path", type=str, default='/home/zhengchujie/wordvector/chinese',
                        help="Directory for pretrained wordvector. Default: resources://Glove300d")
    parser.add_argument("--embedding_size", type=int, default=200,
                        help="The embed dim of the pretrained word vector.")

    parser.add_argument("--max_sent_length", default=192, tpye=int,
                        help="The max length of the sentence pair.")
    parser.add_argument("--max_know_length", default=100, type=int,
                        help="The max length of the knowledge triplets.")
    parser.add_argument("--num_turns", default=8, type=int,
                        help="The max turn length of the post field.")
    parser.add_argument("is_relative", action="store_true",
                        help="If True, use relative turn embedding, else use absolute turn embedding.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--cache", action="store_ture", help="Whether to save the data result.")

    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=16, type=int, help="Total batch size for prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for the optimizer.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")

    # ----------------- 用于推理阶段的一些参数 -------------------
    parser.add_argument("--no_sample", action="store_true", help="Set to use greedy decoding instead of sampling.")
    parser.add_argument("--min_decoder_length", type=int, default=3, help="The minimum length of the generated response.")
    parser.add_argument("--max_decoder_length", type=int, default=30, help="The maximum length of the generated response.")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling softmax temperature.")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.0,
                        help="Nucleus filetring (top-p) before sampling (<=0.0: no filtering)")
    # --------------------------------------------------------

    # 表示训练的前多少进行warm_up
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1=10% "
                             "of training.")
    parser.add_argument("--no_cuda", default=False, action="store_true",
                        help="Whether not to use CUDA when available.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case", default=True, action="store_true",
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")

    args = parser.parse_args()


    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir, exist_ok=True)


    data_class = GPTGenKA
    wordvec_class = TencentChinese

    # 加载数据
    def load_dataset(file_id, vocab_name, do_lower_case, max_sent_length, max_know_length, num_turns, is_relative,):
        dm = data_class(file_id=file_id,
                        vocab_name=vocab_name,
                        do_lower_case=do_lower_case,
                        max_sent_length=max_sent_length,
                        max_know_length=max_know_length,
                        num_turns=num_turns,
                        is_relative=is_relative)
        return dm

    logger.info("模型训练侧加载数据")
    if args.cache:
        if not os.path.isdir(args.cache_dir):
            os.mkdir(args.cache_dir)
        dataManager = try_cache(load_dataset,
                                {"file_id": args.datapath,
                                 "vocab_name": args.vocab_file,
                                 "do_lower_case": args.do_lower_case,
                                 "max_sent_length": args.max_sent_length,
                                 "max_know_length": args.max_know_length,
                                 "num_turns": args.num_turns,
                                 "is_relative":args.is_relative},
                                args.cache_dir,
                                data_class.__name__)
        vocab = dataManager.id2know_word
        logger.info("加载词向量文件")
        embed = try_cache(lambda wv, ez, vl: wordvec_class(wv).load_matrix(ez, vl),
                          (args.wv_path, args.embedding_size, vocab),
                          args.cache_dir,
                          wordvec_class.__name__)
































