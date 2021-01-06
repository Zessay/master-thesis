from __future__ import print_function
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import warnings
warnings.filterwarnings("ignore")

import argparse
import logging
import json
import random
from tqdm import trange

import numpy as np
import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from transformers import AdamW

from myCoTK.dataloader import MyMemBERTRetrieval
from myCoTK.wordvector import TencentChinese
from utils.cache_helper import try_cache
from utils.common import seed_everything, save_losses
from utils.MyMetrics import MyMetrics


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_kg_mask(length_tensor, kg_len):
    batch_size = length_tensor.shape[0]
    # [batch_size, max_kg_length, kg_length]
    length_mask_onehot = torch.nn.functional.one_hot(length_tensor, num_classes=kg_len + 1)[:, :, 1:].float()

    cumsum = torch.cumsum(length_mask_onehot, dim=2)
    length_ones = torch.ones_like(length_mask_onehot, dtype=torch.float32)
    origin_mask = length_ones - cumsum + length_mask_onehot
    return origin_mask.view(batch_size, -1, kg_len, 1)


# pylint: disable=W0221
class BERTRetrieval(BertPreTrainedModel):
    def __init__(self, num_choices, bert_config_file, init_embeddings):
        self.num_choices = num_choices
        self.bert_config = BertConfig.from_json_file(bert_config_file)
        BertPreTrainedModel.__init__(self, self.bert_config)

        self.bert = BertModel(self.bert_config)
        self.init_weights()    # 初始化权重参数
        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)

        # 用于知识表征的词向量矩阵
        self.vocab_size, self.embed_size = np.shape(init_embeddings)
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(init_embeddings), freeze=False)

        #self.classifier = nn.Linear(self.bert_config.hidden_size + self.embed_size, 1)
        self.classifier = nn.Linear(self.bert_config.hidden_size, 1)
        self.reshape = nn.Linear(self.bert_config.hidden_size, self.embed_size, bias=False)
        self.reshape_know = nn.Linear(self.embed_size, self.bert_config.hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.Sigmoid()

    def forward(self, data, labels=None):
        input_ids = data['input_ids']
        token_type_ids = data['segment_ids']
        attention_mask = data['input_mask']
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pair_output = outputs[1]
        if labels is not None:
            pair_output = self.dropout(pair_output)

        # 这里的batch_size相当于指定的train_batch_size * num_choices
        # data['kg]的形状是batch_size * max_kg_num * max_kg_hrt_length
        # max_kg_num表示当前batch中一段对话中最大的kg的数量
        # max_kg_hrt_length表示当前batch中
        batch_size = data['kg'].shape[0]
        kg_len = data['kg'].shape[2]

        # length对应的维度为[batch_size, max_kg_length]
        # 根据长度计算知识对应的mask矩阵
        # 输出维度为[batch_size, max_kg_num, max_kg_hrt_length, 1]
        kg_hr_mask = get_kg_mask(data['kg_hr_length'], kg_len)
        kg_hrt_mask = get_kg_mask(data['kg_hrt_length'], kg_len)
        kg_key_mask = kg_hr_mask                  # 这个mask只关注head entity和relation
        kg_value_mask = kg_hrt_mask - kg_hr_mask  # 这个mask只关注tail entity

        # 将kg输入的单词转化为embedding
        # [batch_size, max_kg_num, max_kg_hrt_length, embed_dim]
        kg_input = self.embed(data['kg'])
        # 这里将每一个样例中对应的hrt的嵌入相加，相当于得到当前每一个知识的表示
        # 这里只是将head entity和relation的嵌入相加
        # 然后除以对应的长度，相当于词向量的平均
        # [batch_size, max_kg_num, embed_dim]
        kg_key_avg = torch.sum(kg_input * kg_key_mask, dim=2) / torch.max(
            torch.sum(kg_key_mask, dim=2), torch.ones_like(data['kg_hrt_length'].unsqueeze(-1), dtype=torch.float32))
        kg_value_avg = torch.sum(kg_input * kg_value_mask, dim=2) / torch.max(
            torch.sum(kg_value_mask, dim=2), torch.ones_like(data['kg_hrt_length'].unsqueeze(-1), dtype=torch.float32))
        # 这里相当于是句对经过BERT之后的表征
        # 首先经过线性层将bert表征的维度转化为和单词相同的维度
        # [batch_size, 1, embed_dim]
        query = torch.reshape(self.reshape(pair_output), [batch_size, 1, self.embed_size])
        # 根据cls的向量和知识中的head和relation平均向量，计算attention的分数
        # [batch_size, max_kg_num]
        kg_score = torch.sum(query * kg_key_avg, dim=2)
        # 这里根据hrt的长度，计算kg的mask向量
        # [batch_size, max_kg_num]
        cond = (data['kg_hrt_length'] > 0).float()
        # 将mask的位置设为负无穷
        kg_score = kg_score * cond - (torch.ones_like(cond, dtype=torch.float32) - cond) * 1e23
        # 得到对于每个知识的attention分数
        # [batch_size, max_kg_num]
        kg_alignment = self.softmax(kg_score)

        # 找到attention分数最大的位置
        kg_max = torch.argmax(kg_alignment, dim=-1)
        # 将对应位置设为1
        # [batch_size, max_kg_num]
        kg_max_onehot = torch.nn.functional.one_hot(kg_max, num_classes=kg_alignment.shape[1]).float()

        # 计算每一个样例的加权平均向量
        # [batch_size, embed_dim]
        knowledge_embed = torch.sum(kg_alignment.unsqueeze(-1) * kg_value_avg, dim=1)
        # 将知识的表征embed_dim重新映射到BERT的表征维度 bert_hidden_size
        # [batch_size, bert_hidden_size]
        relu_know = self.relu(self.reshape_know(knowledge_embed))
        # 经过分类器分类，并转化为概率
        logits = self.classifier(pair_output + relu_know)
        prob = self.activation(logits.view(-1))

        if labels is not None:
            # 计算选择的知识和实际使用的知识匹配的数量总和
            kg_acc_num = torch.sum(torch.sum(data['kg_index'] * kg_max_onehot, dim=1) * labels)
            # 计算总计使用的知识数量
            kg_all_num = torch.sum(torch.max(data['kg_index'], dim=1)[0] * labels)
            kg_acc = kg_acc_num / torch.max(kg_all_num, torch.FloatTensor([1]).cuda())

            # 计算关注正确的kg的损失
            # 这里data['kg_index']是mask类型的向量[batch_size, max_kg_num]，真实使用的知识对应的位置为1
            kg_loss = self.num_choices * torch.mean(
                labels * torch.sum(-torch.log(torch.clamp(kg_alignment, 1e-12, 1.0)) * data['kg_index'], dim=1) / torch.max(
                    torch.sum(data['kg_index'], dim=1), torch.ones([batch_size], dtype=torch.float32).cuda()))

            loss = torch.mean(-labels * torch.log(prob) - (1 - labels) * torch.log(1 - prob))
            return loss, kg_loss, kg_acc
        else:
            prob = prob.view(-1, self.num_choices)
            pred = torch.argmax(prob.view(-1, self.num_choices), dim=1)
            return prob.cpu().numpy(), pred


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def preprocess_batch(data, device=None):
    for key in ['input_ids', 'input_mask', 'segment_ids', 'labels', 'kg_hr_length', 'kg_hrt_length', 'kg', 'kg_index']:
        if key != 'labels' and key != 'kg_index':
            data[key] = torch.LongTensor(data[key])
        else:
            data[key] = torch.FloatTensor(data[key])

        if device is not None:
            data[key] = data[key].to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_config_file", default="chinese_wwm_pytorch/bert_config.json",
                        type=str, help="The config json file corresponding to the pre-trained BERT model. "
                                       "This specifies the model architecture.")
    parser.add_argument("--vocab_file", default="chinese_wwm_pytorch/vocab.txt", type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--init_checkpoint", default="chinese_wwm_pytorch/pytorch_model.bin",
                        type=str, help="Initial checkpoint (usually from a pre-trained BERT model).")

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--model_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--cache_dir", default=None, type=str, required=True, help="Whether to run training.")

    ## Other parameters
    parser.add_argument('--name', type=str, default='BERTRetrieval', help='name of model')

    parser.add_argument('--dataset', type=str, default='ChDialogMemBERTRetrieval', help='Dataloader class. Default: OpenSubtitles')
    parser.add_argument('--datapath', type=str, default='resources://OpenSubtitles',
                        help='Directory for data set. Default: resources://OpenSubtitles')
    parser.add_argument('--wv_class', type=str, default='TencentChinese',
                        help="Wordvector class, none for not using pretrained wordvec. Default: Glove")
    parser.add_argument('--wv_path', type=str, default='/home/zhengchujie/wordvector/chinese',
                        help="Directory for pretrained wordvector. Default: resources://Glove300d")
    parser.add_argument('--embedding_size', type=int, default=200,
                        help="Directory for pretrained wordvector. Default: resources://Glove300d")

    parser.add_argument("--num_choices", default=10, type=int,
                        help="the number of retrieval options")
    parser.add_argument("--max_sent_length", default=192, type=int,
                        help="The max length of the sentence pair.")
    parser.add_argument("--num_turns", default=8, type=int,
                        help="The max turn length of the post field.")

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--cache", action='store_true', help="Whether to run training.")

    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=16, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument("--lamb", default=0.6, type=float,
                        help="The factor of the attention loss.")

    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")

    args = parser.parse_args()

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir, exist_ok=True)

    data_class = MyMemBERTRetrieval
    wordvec_class = TencentChinese

    # 加载数据
    def load_dataset(file_id, bert_vocab_name, do_lower_case, num_choices, max_sent_length, num_turns):
        dm = data_class(file_id=file_id, bert_vocab_name=bert_vocab_name, do_lower_case=do_lower_case, num_choices=num_choices,
                        max_sent_length=max_sent_length, num_turns=num_turns)
        return dm

    logger.info("模型训练侧加载数据")
    if args.cache:
        if not os.path.isdir(args.cache_dir):
            os.mkdir(args.cache_dir)
        dataManager = try_cache(load_dataset,
                                (args.datapath, args.vocab_file, args.do_lower_case, args.num_choices, args.max_sent_length, args.num_turns),
                                args.cache_dir,
                                data_class.__name__)
        vocab = dataManager.id2know_word
        logger.info("加载词向量文件")
        embed = try_cache(lambda wv, ez, vl: wordvec_class(wv).load_matrix(ez, vl), (args.wv_path, args.embedding_size, vocab),
                          args.cache_dir, wordvec_class.__name__)
    else:
        dataManager = load_dataset(file_id=args.datapath, bert_vocab_name=args.vocab_file, do_lower_case=args.do_lower_case,
                                   num_choices=args.num_choices, max_sent_length=args.max_sent_length,
                                   num_turns=args.num_turns)
        logger.info("加载词向量文件")
        wv = wordvec_class(args.wv_path)
        vocab = dataManager.id2know_word
        embed = wv.load_matrix(args.embedding_size, vocab)

    #dataManager._max_know_length = 100

    if args.do_train:

        if not args.no_cuda:
            if not "CUDA_VISIBLE_DEVICES" in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        logger.info("device: {} n_gpu: {}".format(device, n_gpu))

        if args.gradient_accumulation_steps < 1:
            raise ValueError(
                "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

        args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

        seed_everything(args.seed)
        if n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)


        logger.info("train examples {}".format(len(dataManager.data['train']['resp'])))
        num_train_steps = int(len(dataManager.data['train'][
                                      'resp']) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

        # Prepare model
        '''
        if os.path.exists(output_model_file):
            model_state_dict = torch.load(output_model_file)
            model = BERTRetrieval(num_choices=args.num_choices, bert_config_file=args.bert_config_file)
            model.load_state_dict(model_state_dict)
        '''
        model = BERTRetrieval(num_choices=args.num_choices, bert_config_file=args.bert_config_file, init_embeddings=embed)
        if args.init_checkpoint is not None:
            logger.info('load bert weight')
            state_dict = torch.load(args.init_checkpoint, map_location='cpu')
            missing_keys = []
            unexpected_keys = []
            error_msgs = []
            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, '_metadata', None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            def load(module, prefix=''):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

                module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys,
                                             error_msgs)
                for name, child in module._modules.items():
                    # logger.info("name {} chile {}".format(name,child))
                    if child is not None:
                        load(child, prefix + name + '.')

            load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
            logger.info("missing keys:{}".format(missing_keys))
            logger.info('unexpected keys:{}'.format(unexpected_keys))
            logger.info('error msgs:{}'.format(error_msgs))

        model.to(device)
        model = torch.nn.DataParallel(model)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        t_total = num_train_steps
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num post-response pairs = %d", len(dataManager.data['train']['resp']))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        model.train()
        losses = []
        kg_losses = []
        kg_accs = []
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.zero_grad()
            dataManager.restart(key='train', batch_size=args.train_batch_size)
            data = dataManager.get_next_batch(key='train')
            step = 0
            while data is not None:
                if n_gpu == 1:
                    preprocess_batch(data, device) # multi-gpu does scattering it-self
                else:
                    preprocess_batch(data)
                loss, kg_loss, kg_acc = model(data, data['labels'])
                loss = loss + args.lamb * kg_loss
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # 每次反向传播之前记录一下当前的指标
                    losses.append(loss.detach().cpu().item())
                    kg_losses.append(kg_loss.detach().cpu().item())
                    kg_accs.append(kg_acc.detach().cpu().item())



                if (step + 1) % 1000 == 0:
                    logger.info("step:{} | loss@{} | kg_loss@{} | kg_acc@{}".format(step + 1,
                                                                                    loss.cpu().item(),
                                                                                    kg_loss.cpu().item(),
                                                                                    kg_acc.cpu().item()))

                step += 1
                data = dataManager.get_next_batch(key='train')

            logger.info(f"保存模型 pytorch_model.{int(args.num_train_epochs)}.{epoch+1}.bin")
            output_model_file = os.path.join(args.model_dir,
                                             "pytorch_model.%d.%d.bin" % (int(args.num_train_epochs), epoch + 1))

            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), output_model_file)

        # 保存所有的损失值
        logger.info("保存训练过程的loss")
        save_losses(args.model_dir, losses={"loss": losses,
                                            "kg_loss": kg_losses,
                                            "kg_acc": kg_accs})
        logger.info("训练结束")
    # Load a trained model that you have fine-tuned

    if args.do_predict:

        test_distractors = dataManager.data['test']['resp_distractors']
        with open(os.path.join(args.datapath, 'test_distractors.json'), 'w') as f:
            json.dump(test_distractors, f, ensure_ascii=False, indent=4)

        if not args.no_cuda:
            if not "CUDA_VISIBLE_DEVICES" in os.environ:
                os.environ["CUDA_VISIBLE_DEVICES"] = '2'

        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()

        output_model_file = os.path.join(args.model_dir, "pytorch_model.%d.%d.bin" %
                                         (int(args.num_train_epochs),
                                          int(args.num_train_epochs)
                                          ))

        model_state_dict = torch.load(output_model_file)
        model = BERTRetrieval(num_choices=args.num_choices, bert_config_file=args.bert_config_file, init_embeddings=embed)
        model.load_state_dict(model_state_dict)
        model.to(device)

        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        metric = MyMetrics()

        logger.info("***** Running testing *****")
        logger.info("  Num post-response pairs = %d", len(dataManager.data['test']['resp']))
        logger.info("  Batch size = %d", args.predict_batch_size)

        model.eval()
        logger.info("Start evaluating")
        dataManager.restart(key='test', batch_size=args.predict_batch_size, shuffle=False)
        data = dataManager.get_next_batch(key='test')

        gens = []
        gold = []
        choices = []

        hits = {1: [0, 0], 3:[0, 0], 5: [0, 0]}
        while data is not None:
            preprocess_batch(data, device)
            truth_response, can_responses = data['resp'], data['can_resps']

            with torch.no_grad():
                prob, pred = model(data)

            assert len(pred) == len(truth_response)
            assert len(pred) == len(can_responses)
            assert len(can_responses[0]) == args.num_choices

            for truth, pd, cans, prb in zip(truth_response, pred, can_responses, prob):
                metric.forword(truth, cans[pd])

                gold.append(truth)
                gens.append(cans[pd])
                choices.append(cans)

                idx = cans.index(truth)
                p_sort = np.argsort(prb)
                for key, count in hits.items():
                    if idx in p_sort[-key:]:
                        count[0] += 1
                    count[1] += 1

            data = dataManager.get_next_batch(key='test')

        result = metric.close()
        result.update({'hits@%d' % key: value[0] / value[1] for key, value in hits.items()})

        output_prediction_file = args.output_dir + "/%s_%s.txt" % (args.name, "test")
        with open(output_prediction_file, "w") as f:
            print("Test Result:")
            res_print = list(result.items())
            res_print.sort(key=lambda x: x[0])
            for key, value in res_print:
                if isinstance(value, float):
                    print("\t%s:\t%f" % (key, value))
                    f.write("%s:\t%f\n" % (key, value))
            f.write('\n')

            for resp, gen, options in zip(gold, gens, choices):
                f.write("resp:\t%s\n" % resp)
                f.write("gen:\t%s\n\n" % gen)
                for i, option in enumerate(options):
                    f.write("candidate %d:\t%s\n" % (i, option))
                f.write("\n")


if __name__ == "__main__":
    main()