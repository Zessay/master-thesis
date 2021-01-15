# coding=utf-8
# @Author: 莫冉
# @Date: 2021-01-11
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
import torch.nn.functional as F
import torch.nn as nn
from itertools import chain
from transformers import BertTokenizer, GPT2Config
from transformers import AdamW

from myModels.modeling_gpt2 import GPT2LMHeadModel
from myCoTK.bert_dataloader import GPTGen
from utils.cache_helper import try_cache
from utils.common import seed_everything, save_losses
from utils.MyMetrics import MyMetrics, MyPerplexity


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[speaker1]", "[speaker2]"]


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def preprocess_batch(data, device=None):
    for key in ['input_ids', 'input_mask', 'token_type_ids', 'turn_ids', 'source_ids', 'lm_labels']:
        if "labels" not in key:
            data[key] = torch.LongTensor(data[key])
        else:
            data[key] = torch.FloatTensor(data[key])

        if device is not None:
            data[key] = data[key].to(device)


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def build_input_from_segments(history, reply, special_token_ids, token_type_ids=None, turn_ids=None, with_eos=True):
    """
    history: List[int], 这里的history是测试集经过处理之后的，包含了历史的对话，添加了对话人
    reply: List[int]，这里reply表示之前解码的结果
    """
    bos, eos, pad, speaker1, speaker2 = special_token_ids
    sequence = history + reply + [eos] if with_eos else []
    instance = {}
    instance["input_ids"] = list(sequence)
    if token_type_ids is not None:
        extend_token_type_ids = [token_type_ids[-1]] * ((len(reply) + 1) if with_eos else len(reply))
        instance["token_type_ids"] = token_type_ids.extend(extend_token_type_ids)
    else:
        instance["token_type_ids"] = None

    if turn_ids is not None:
        extend_turn_ids = [turn_ids[-1]] * ((len(reply) + 1) if with_eos else len(reply))
        instance["turn_ids"] = turn_ids.extend(extend_turn_ids)
    else:
        instance["turn_ids"] = None
    return instance, sequence


def sample_sequence(history, model, args, device, special_tokens_ids,
                    token_type_ids=None, turn_ids=None, current_output=None):
    bos, eos , *_ = special_tokens_ids
    if current_output is None:
        current_output = []

    sent_logits = []
    for i in range(args.max_decoder_length):
        instance, sequence = build_input_from_segments(history,
                                                       current_output,
                                                       special_tokens_ids,
                                                       token_type_ids=token_type_ids,
                                                       turn_ids=turn_ids,
                                                       with_eos=False)
        input_ids = torch.tensor(instance["input_ids"], dtype=torch.long, device=device).unsqueeze(0)
        if instance["token_type_ids"] is not None:
            token_type_ids = torch.tensor(instance["token_type_ids"], dtype=torch.long, device=device).unsqueeze(0)
        else:
            token_type_ids = None
        if instance["turn_ids"] is not None:
            turn_ids = torch.tensor(instance["turn_ids"], dtype=torch.long, device=device).unsqueeze(0)
        else:
            turn_ids = None

        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, turn_ids=turn_ids)
        logits = outputs["logits"]
        # 获取最后一个单词的概率分布(vocab_size, )
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        # 决定是根据概率分布进行多项式采样，还是直接贪婪搜索
        # 这里直接得到单词对应的索引
        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_decoder_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() == eos:
            break
        current_output.append(prev.item())
        sent_logits.append(torch.log(probs))
    return current_output, sent_logits


def test_process_hits(data, model, args):
    with open(os.path.join(args.datapath, 'test_distractors.json'), 'r', encoding="utf-8") as f:
        test_distractors = json.load(f)

    data.restart("test", batch_size=1, shuffle=False)
    batched_data = data.get_next_batch("test")

    loss_record = []
    cnt = 0
    while batched_data != None:
        for key in batched_data:
            if isinstance(batched_data[key], np.ndarray):
                batched_data[key] = batched_data[key].tolist()

        resp_ids = []     # 保存所有回复对应的转换之后的id
        resp_lens = []    # 保存所有回复对应的长度
        # 取出当前样本对应的负样本回复
        # 当获取test数据时，input_ids对应 post + response的对话人角色
        # input_mask对应空列表
        # token_type_ids和input_ids类似
        # turn_ids和token_type_ids一致
        # posts_len则不包含response对应的对话人id，只是处理之后的posts的长度
        gt_resp = batched_data["resp"][0]  # 表示resp的单词列表，List[str]
        resp_ids.append(data.convert_tokens_to_bert_ids(gt_resp) + [data.bert_eos_id])
        resp_lens.append(batched_data["resp_lens"][0] + 1)
        for each_resp in test_distractors[cnt]:
            resp_ids.append(data.convert_tokens_to_bert_ids(data.tokenize(each_resp)) + [data.bert_eos_id])
            resp_lens.append(len(resp_ids[-1]))

        # 这里加上1为了考虑dataManager中对回复添加的对话人角色
        max_length = batched_data["posts_lens"][0] + 1 + max(resp_lens)
        origin_input_ids = batched_data["input_ids"][0]
        origin_token_type_ids = batched_data["token_type_ids"][0]
        origin_turn_ids = batched_data["turn_ids"][0]
        input_ids, input_mask, token_type_ids, turn_ids, lm_labels = [], [], [], [], []
        # 对于每一个回复，转化为GPT的输入
        for resp_id, resp_len in zip(resp_ids, resp_lens):
            new_input_ids = copy.deepcopy(origin_input_ids)
            new_token_type_ids = copy.deepcopy(origin_token_type_ids)
            new_turn_ids = copy.deepcopy(origin_turn_ids)
            posts_len = batched_data["posts_lens"][0]

            # 将处理之后的回复id添加到new_input_ids中
            new_input_ids += resp_id
            new_token_type_ids += [new_token_type_ids[-1]] * resp_len
            new_turn_ids += [new_turn_ids[-1]] * resp_len
            new_input_mask = [1] * len(new_input_ids)

            # 计算标签
            new_lm_labels = [-1] * (posts_len) + resp_id   # 计算loss时会有shift操作，所以这里只需要逐一对齐即可

            padding = [0] * (max_length - len(input_ids))
            new_input_ids += padding
            new_token_type_ids += padding
            new_turn_ids += padding
            new_input_mask += padding
            new_lm_labels += [-1] * len(padding)

            # 添加到最终的输入中
            input_ids.append(new_input_ids)
            input_mask.append(new_input_mask)
            token_type_ids.append(new_token_type_ids)
            turn_ids.append(new_turn_ids)
            lm_labels.append(new_lm_labels)

        input_ids = torch.tensor(input_ids, device=model.device, dtype=torch.long)
        input_mask = torch.tensor(input_mask, device=model.device, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, device=model.device, dtype=torch.long)
        turn_ids = torch.tensor(turn_ids, device=model.device, dtype=torch.long)
        lm_labels = torch.tensor(lm_labels, device=model.device, dtype=torch.float32)
        outputs = model(input_ids=input_ids,
                        attention_mask=input_mask,
                        token_type_ids=token_type_ids,
                        turn_ids=turn_ids,
                        labels=lm_labels,
                        reduction='none')
        batch_loss = outputs["loss"]
        loss_record.append(batch_loss.detach().cpu().numpy().tolist())
        cnt += 1
        batched_data = data.get_next_batch("test")

    assert cnt == len(test_distractors)

    loss = np.array(loss_record)
    loss_rank = np.argsort(loss, axis=1)
    hits1 = float(np.mean(loss_rank[:, 0] == 0))
    hits3 = float(np.mean(np.min(loss_rank[:, :3], axis=1) == 0))
    hits5 = float(np.mean(np.min(loss_rank[:, :5], axis=1) == 0))
    return {'hits@1': hits1, 'hits@3': hits3, 'hits@5': hits5}




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
    parser.add_argument("--max_sent_length", default=192, tpye=int,
                        help="The max length of the sentence pair.")
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

    data_class = GPTGen
    config_class = GPT2Config
    model_class = GPT2LMHeadModel
    tokenizer_class = BertTokenizer


    # 加载数据
    def load_dataset(file_id, vocab_name, do_lower_case, max_sent_length, num_turns, is_relative):
        dm = data_class(file_id=file_id, vocab_name=vocab_name, do_lower_case=do_lower_case,
                        max_sent_length=max_sent_length, num_turns=num_turns, is_relative=is_relative)
        return dm

    logger.info("模型训练侧加载数据")
    if args.cache:
        dataManager = try_cache(load_dataset,
                                {"file_id": args.datapath,
                                 "vocab_name": args.vocab_file,
                                 "do_lower_case": args.do_lower_case,
                                 "max_sent_length": args.max_sent_length,
                                 "num_turns": args.num_turns,
                                 "is_relative": args.is_relative},
                                args.cache_dir,
                                data_class.__name__)
    else:
        dataManager = load_dataset(file_id=args.datapath,
                                   vocab_name=args.vocab_file,
                                   do_lower_case=args.do_lower_case,
                                   max_sent_length=args.max_sent_length,
                                   num_turns=args.num_turns,
                                   is_relative=args.is_relative)

    if args.do_train:
        if not args.no_cuda:
            if not "CUDA_VISIBLE_DEVICES" in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        logger.info(f"device: {device} | n_gpu: {n_gpu}")

        if args.gradient_accumulation_steps < 1:
            raise ValueError(
                f"Invalid gradient_accumulation_steps parameter: {args.gradient_accumulation_steps}, should be >= 1")

        args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

        seed_everything(args.seed)

        logger.info("train examples {}".format(len(dataManager.data["train"]["resp"])))
        num_train_steps = int(len(dataManager.data["train"]["resp"]) \
                              / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        # 加载预训练模型
        config = config_class.from_json_file(args.gpt_config_file)
        config.num_turns = args.num_turns
        model = model_class(config)
        if args.init_checkpoint is not None:
            logger.info("加载GPT预训练权重")
            state_dict = torch.load(args.init_checkpoint, map_location="cpu")
            missing_keys = []
            unexpected_keys = []
            error_msgs = []
            # 深拷贝state_dict，便于下面的_load_from_state_dict进行修改
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            def load(module, prefix=""):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys,
                                             unexpected_keys, error_msgs)
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + ".")

            load(model, prefix="" if hasattr(model, "transformer") else "transformer.")
            logger.info("missing keys: {}".format(missing_keys))
            logger.info("unexpected keys: {}".format(unexpected_keys))
            logger.info("error msgs: {}".format(error_msgs))

        model.to(device)
        model = torch.nn.DataParallel(model)

        # 准备优化器和优化参数
        param_optimizer = list(model.named_parameters())

        # 去除pooling层，这一层会产生梯度None
        # 影响apex的使用
        param_optimizer = [n for n in param_optimizer if "pooler" not in n[0]]
        no_decay = ["bias", "ln_1.bias", "ln_1.weight", "ln_2.bias", "ln_2.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]

        t_total = num_train_steps
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num training_examples = %d", len(dataManager.data['train']['resp']))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        model.train()
        losses = []
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.zero_grad()
            # 初始化数据
            dataManager.restart(key="train", batch_size=args.train_batch_size, shuffle=True)
            # 获取下一个batch的数据
            data = dataManager.get_next_batch(key="train")
            step = 0
            loss_value = 0
            while data is not None:
                if n_gpu == 1:
                    preprocess_batch(data, device)
                else:
                    preprocess_batch(data)

                outputs = model(input_ids=data["input_ids"],
                                attention_mask=data["input_mask"],
                                token_type_ids=data["token_type_ids"],
                                turn_ids=data["turn_ids"],
                                labels=data["lm_labels"])
                # loss: 是一个tensor型的标量
                # lm_logits: [batch, seq_length, vocab_size]
                loss, lm_logits = outputs[0], outputs[1]
                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss_value += loss.cpu().item() * args.gradient_accumulation_steps
                loss.backward()
                # 如果达到累积的梯度数量，则反向传播
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify leanring rate with special warm up GPT
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    losses.append(loss.detach().cpu().item())

                # 如果要记录当前的损失值
                if (step + 1) % 1000 == 0:
                    logger.info(f"step: {step + 1} | loss: {(loss_value / 1000):.4f} | ppl: {(np.exp(loss_value / 1000)):.4f}")
                    loss_value = 0

                step += 1
                data = dataManager.get_next_batch(key="train")


            logger.info(f"保存模型 pytorch_model.{int(args.num_train_epochs)}.{epoch+1}.bin")
            output_model_file = os.path.join(args.model_dir, f"pytorch_model.{int(args.num_train_epochs)}.{int(epoch+1)}.bin")

            # 保存训练好的模型
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), output_model_file)

        # 保存损失
        logger.info("保存训练过程中的loss")
        save_losses(args.model_dir, losses={"loss": losses})
        logger.info("训练结束")


    # 定义测试集上运行的过程
    if args.do_predict:

        total_epoch = int(args.num_train_epochs)
        chosen_epoch = 4

        if not args.no_cuda:
            if not "CUDA_VISIBLE_DEVICES" in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()

        seed_everything(args.seed)

        output_model_file = os.path.join(args.model_dir, "pytorch_model.%d.%d.bin" % (total_epoch, chosen_epoch))
        model_state_dict = torch.load(output_model_file)

        tokenizer = tokenizer_class(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
        config = config_class.from_json_file(args.gpt_config_file)
        config.num_turns = args.num_turns
        model = model_class(config)
        model.load_state_dict(model_state_dict)
        model.to(device)

        logger.info(f"transform special tokens {SPECIAL_TOKENS} to ids")
        special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # 这里的myMetrics主要用来计算bleu和distinct
        # 这里bleu和distinct计算是按照中文word计算的，而不是按照char
        metric1 = MyMetrics()
        metric2 = MyPerplexity(unk_id=dataManager.bert_unk_id)

        logger.info("***** Running testing *****")
        logger.info("  Num post-response pairs = %d", len(dataManager.data['test']['resp']))
        logger.info("  Batch size = %d", args.predict_batch_size)

        model.eval()
        logger.info("Start evaluating")
        dataManager.restart(key='test', batch_size=args.predict_batch_size, shuffle=False)
        data = dataManager.get_next_batch(key='test')

        # 保存预测和生成的结果
        gold_strings = []
        gen_strings = []

        while data is not None:
            cur_batch_size = int(len(data["input_ids"]))
            for i in range(cur_batch_size):
                input_ids = data["input_ids"][i]
                token_type_ids = data["token_type_ids"][i]
                turn_ids = data["turn_ids"][i]
                # posts_len = data["posts_len"][i]
                resp_list = data["resp"][i]      # 这里是tokenizer分词之后的列表，并没有转化为id
                resp_length = data["resp_lens"][i]

                # 这里得到最终输出的所有ids
                with torch.no_grad():
                    # 这里的pred_logits是经过log_softmax之后的
                    # pred_ids: [seq_len]
                    # pred_logits: [seq_len, vocab_size], torch.Tensor类型
                    pred_ids, pred_logits = sample_sequence(history=input_ids,
                                                            model=model,
                                                            args=args,
                                                            device=device,
                                                            special_tokens_ids=special_tokens_ids,
                                                            token_type_ids=token_type_ids,
                                                            turn_ids=turn_ids,
                                                            current_output=None)
                # 将输出的ids转化为tokens
                # decode的输出是一个字符串，token之间使用空格拼接
                pred_text = tokenizer.decode(pred_ids, skip_special_tokens=False)
                # 计算bleu和distinct指标
                pred_text_string = "".join(pred_text.split())
                resp_text_string = "".join(resp_list)
                metric1.forward(ref=resp_text_string, hyp=pred_text_string)
                # 计算ppl
                # 将resp_list转化为torch.Tensor类型的token_id
                resp_ids = torch.tensor(tokenizer.convert_tokens_to_ids(resp_list),
                                        dtype=torch.long, device=pred_logits.device)
                metric2.forward(resp_length=resp_length, resp_ids=resp_ids, gen_log_prob=pred_logits)
                gold_strings.append(resp_text_string)
                gen_strings.append(pred_text_string)
            data = dataManager.get_next_batch(key="test")

        hits = test_process_hits(dataManager, model, args)
        result = metric1.close()
        result.update(metric2.close())
        result.update(hits)

        # 保存预测结果
        output_prediction_file = args.output_dir + f"/{args.name}_test.{total_epoch}.{chosen_epoch}.txt"
        logger.info(f"预测指标保存的路径 {output_prediction_file}")
        with open(output_prediction_file, "w", encoding="utf-8") as f:
            print("Test Result: ")
            res_print = list(result.items())
            res_print.sort(key=lambda x: x[0])
            for key, value in res_print:
                if isinstance(value, float):
                    print(f"\t{key}:\t{value}")
                    f.write(f"{key}:\t{value}\n")
            f.write("\n")

            for gold, gen in zip(gold_strings, gen_strings):
                f.write(f"resp:\t{gold}\n")
                f.write(f"gen:\t{gen}\n\n")


if __name__ == '__main__':
    main()









