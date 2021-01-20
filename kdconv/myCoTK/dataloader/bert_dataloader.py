'''
A module for BERT dataloader
'''
from cotk.dataloader import LanguageProcessingBase
from cotk._utils import trim_before_target
import numpy as np
import os
import multiprocessing
import logging
import time
from itertools import chain
import json
import random

from cotk.metric import MetricChain, SingleTurnDialogRecorder, PerplexityMetric
from ..metric import BleuCorpusMetric, SingleTurnDistinct, SingleTurnResponseRecorder
from transformers import BertTokenizer
# from pytorch_pretrained_bert.tokenization import BertTokenizer

import jieba
from gensim.summarization import bm25
from typing import List, Optional

from utils.lcs import find_lcs, find_substring_pos_pair


logger = logging.getLogger(__name__)

# -----------------------------------------------------------------

# 定义用于BERT的基类
class BERTLanguageProcessingBase(LanguageProcessingBase):
	r"""Base class for all BERT-based language processing with BERT tokenizer.
	This is an abstract class.

	Arguments:{ARGUMENTS}

	Attributes:{ATTRIBUTES}
	"""

	BERT_VOCAB_NAME = r"""
			bert_vocab_name (str): A string indicates which bert model is used, it will be a
					parameter passed to `pytorch-transformers.BertTokenizer.from_pretrained
					<https://github.com/huggingface/pytorch-transformers#berttokenizer>`_.
					It can be 'bert-[base|large]-[uncased|cased]' or a local path."""

	ARGUMENTS = LanguageProcessingBase.ARGUMENTS + BERT_VOCAB_NAME

	ATTRIBUTES = LanguageProcessingBase.ATTRIBUTES + r"""
			bert_id2word (list): Vocabulary list mapping bert ids to tokens,
					including valid vocabs and invalid vocabs.
			word2bert_id (dict): A dict mapping all tokens to its bert id. You don't need to use it 
					at most times, see :meth:`convert_tokens_to_bert_ids` instead.
	"""

	def __init__(self, ext_vocab=None, \
					key_name=None, \
					bert_vocab_name='bert-base-uncased'):

		# initialize by default value. (can be overwritten by subclass)
		self.ext_vocab = ext_vocab or ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]

		self.tokenizer = BertTokenizer.from_pretrained(bert_vocab_name)
		self._build_bert_vocab()

		super().__init__(self.ext_vocab, key_name)

	def _build_bert_vocab(self):
		self.word2bert_id = dict(self.tokenizer.vocab)
		self.bert_id2word = [None] * len(self.word2bert_id)
		for key, value in self.word2bert_id.items():
			self.bert_id2word[value] = key

		self.bert_pad_id = self.word2bert_id["[PAD]"]
		self.bert_unk_id = self.word2bert_id["[UNK]"]
		self.bert_go_id = self.word2bert_id["[CLS]"]
		self.bert_eos_id = self.word2bert_id["[SEP]"]

	def _valid_bert_id_to_id(self, bert_id):
		'''This function return the id for a valid bert id, otherwise return ``unk_id``.

		Arguments:
			bert_id (str): a bert id.

		Returns:
			int
		'''
		idx = self.word2id.get(bert_id, self.unk_id)
		if idx >= self.vocab_size:
			idx = self.unk_id
		return idx

	def tokenize(self, sentence):
		'''Convert sentence(str) to list of tokens(str)

		Arguments:
				sentence (str)

		Returns:
				sent (list): list of tokens(str)
		'''
		return self.tokenizer.tokenize(sentence)

	def convert_tokens_to_bert_ids(self, sent):
		'''Convert list of token(str) to list of bert id(int)

		Arguments:
				sent (list): list of token(str)

		Returns:
				bert_ids (list): list of bert id(int)
		'''
		return self.tokenizer.convert_tokens_to_ids(sent)

	def convert_bert_ids_to_tokens(self, bert_ids, trim=True):
		'''Convert list of bert id(int) to list of token(str)

		Arguments:
				bert_ids (list): list of bert id(int)

		Returns:
				(list): list of token(str)
		'''
		if trim:
			bert_ids = trim_before_target(list(bert_ids), self.bert_eos_id)
			idx = len(bert_ids)
			while idx > 0 and bert_ids[idx-1] == self.bert_pad_id:
				idx -= 1
			bert_ids = bert_ids[:idx]
		return list(map(lambda word: self.bert_id2word[word], bert_ids))

	def convert_bert_ids_to_ids(self, bert_ids, invalid_vocab=False):
		'''Convert list of bert id(int) to list of id(int)

		Arguments:
				bert_ids (list): list of bert id(int)
				invalid_vocab (bool): whether to provide invalid vocabs.
					If ``False``, invalid vocabs will be replaced by ``unk_id``.
					If ``True``, invalid vocabs will using their own id.
					Default: ``False``

		Returns:
				(list): list of id(int)
		'''
		return self.convert_tokens_to_ids(\
			self.convert_bert_ids_to_tokens(bert_ids, False), invalid_vocab)

	def convert_ids_to_bert_ids(self, ids):
		'''Convert list of id(int) to list of bert id(int)

		Arguments:
				ids (list): list of id(int)

		Returns:
				bert_ids (list): list of bert id(int)
		'''
		return self.convert_tokens_to_bert_ids(\
			self.convert_ids_to_tokens(ids, False))

# -----------------------------------------------------------------

class MyBERTRetrieval(BERTLanguageProcessingBase):
    def __init__(self, file_id, bert_vocab_name, do_lower_case, num_choices=10,
                 max_sent_length=192, num_turns=8,
                 ext_vocab=None, key_name=None, cpu_count=None):
        self.ext_vocab = ext_vocab or ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        self._max_sent_length = max_sent_length
        self._file_path = file_id
        self.num_choices = num_choices
        self.num_turns = num_turns
        super().__init__(self.ext_vocab, key_name, bert_vocab_name)

        self.tokenizer = BertTokenizer(vocab_file=bert_vocab_name, do_lower_case=do_lower_case)
        self._build_bert_vocab()

        if cpu_count is not None:
            self.cpu_count = cpu_count
        elif "CPU_COUNT" in os.environ and os.environ["CPU_COUNT"] is not None:
            self.cpu_count = int(os.environ["CPU_COUNT"])
        else:
            self.cpu_count = multiprocessing.cpu_count()

    def _load_data(self):
        r'''Loading dataset, invoked by `BERTLanguageProcessingBase.__init__`
        '''
        logger.info("开始读取和处理数据")
        begin_time = time.time()
        origin_data = {}

        # 读取停用词
        with open("../../data/resources/hit_stopwords.txt", "r", encoding="utf-8") as f:
            stop_words = set([w.strip() for w in f])


        # 逐个文件读取数据
        for key in self.key_name:
            corpus_resp = []
            corpus_post = []

            # post_bert表示当前回复的上文，是一个二维列表，第一维表示上文的轮次，第二维表示每一次的单词数
            # resp表示当前的回复，是str型，每个元素都是一句话
            # resp_bert则是当前回复转化为id之后的一维列表
            origin_data[key] = {'resp': [], 'post_bert': [], 'resp_bert': []}
            datas = json.load(open("%s/%s.json" % (self._file_path, key), "r", encoding="utf-8"))

            logger.info(f"当前正在处理 {key} 的数据，数据量为{len(datas)}")

            # 逐个处理每一组对话
            for data in datas:
                messages = data['messages']
                i = 0

                post_sent = []
                corpus_post_sent = []
                while i + 1 < len(messages):
                    if i == 0:
                        # 将语句使用BERT分词，并转化为对应的id
                        tmp_sent = self.convert_tokens_to_bert_ids(self.tokenize(messages[i]['message']))
                        # 保存了转化之后的id
                        post_sent = [tmp_sent]
                        # 保存语句中的非停用词
                        corpus_post_sent = [
                            [token for token in jieba.lcut(messages[i]['message']) if token not in stop_words]]

                    # 将当前语句的下一句话作为回复，str型
                    origin_data[key]['resp'].append(messages[i + 1]['message'])
                    # 处理之前的回复，去除停用词，只保留有效词
                    sentence = [token for token in jieba.lcut(messages[i + 1]['message']) if token not in stop_words]
                    # 进行分词
                    nxt_sent = self.convert_tokens_to_bert_ids(self.tokenize(messages[i + 1]['message']))

                    # 添加之前的语句作为上下文
                    origin_data[key]['post_bert'].append(post_sent)
                    origin_data[key]['resp_bert'].append(nxt_sent)
                    # 将当前去除停用词之后的语句直接添加到列表中
                    corpus_resp.append(sentence)
                    # 将上文去除停用词之后的语句添加到列表中
                    corpus_post.append(corpus_post_sent)

                    # 将当前回复的上文和当前回复拼接起来，限制上文的最大轮次
                    post_sent = (post_sent + [nxt_sent])[-self.num_turns + 1:]
                    corpus_post_sent = corpus_post_sent + [sentence]

                    i += 1

            logger.info(f"{key}文件共获取有效数据{len(corpus_post)}条")

            logger.info("采样负样本用于训练模型")
            # 当前文件的干扰选项
            distractor_file = os.path.join(self._file_path, '%s_distractors.json' % key)
            # 如果存在当前文件的干扰选项（即负样本）
            if os.path.exists(distractor_file):
                with open(distractor_file, "r", encoding="utf-8") as f:
                    origin_data[key]['resp_distractors'] = json.load(f)
                # 将每一句转化为对应的id
                origin_data[key]['resp_distractors_bert'] = [
                    [self.convert_tokens_to_bert_ids(self.tokenize(sent)) for sent in distractors] for distractors in
                    origin_data[key]['resp_distractors']]
            else:
                # 基于回复构建bm25模型
                # 这里corpus_resp为List[List[str]]，其中每一个str为一个分词之后的单词
                logger.info("构建BM25模型")
                bm25Model = bm25.BM25(corpus_resp)
                # 保存每一次回复的干扰回复
                origin_data[key]['resp_distractors'] = []
                origin_data[key]['resp_distractors_bert'] = []

                logger.info("对有效数据中的每一个样本获取其对应的负样本")
                for idx in range(len(corpus_resp)):
                    # 获取当前回复以及对应的上下文
                    # 都是去除停用词之后的单词列表，posts是多轮的，所以是二维列表
                    posts = corpus_post[idx]
                    resp = corpus_resp[idx]

                    # 计算上文每一个单词的数量
                    count_post_token = {}
                    for token in list(chain(*posts)):
                        if token in count_post_token:
                            count_post_token[token] += 1
                        else:
                            count_post_token[token] = 1

                    # 计算每一个单词的tfidf值
                    token_tfidf = {}
                    for token in count_post_token:
                        # 用当前语句中的频次乘以idf（逆词频）
                        if token in bm25Model.idf:
                            token_tfidf[token] = count_post_token[token] * bm25Model.idf[token]

                    # 对单词排序，tfidf大的靠前，同样大的按照在词表中的顺序排序
                    token_tfidf = sorted(list(token_tfidf.items()), key=lambda x: (-x[1], x[0]))
                    # 获取top5的单词
                    top5 = [each[0] for each in token_tfidf[:5]]

                    # 获取所有回复中对于当前回复和top5单词的BM25分数
                    bm_scores = bm25Model.get_scores(list(set(resp + top5)))
                    bm_scores = np.array(bm_scores)
                    # 对所有回复对于当前回复重要的单词BM25分数进行排序
                    # 即选取和当前回复相似但是又并不正确的回复
                    rank = np.argsort(bm_scores).tolist()
                    # 选择排序最高的前num_choices个回复，作为负样本
                    if idx in rank[-self.num_choices:]:
                        idxs = [each for each in rank[-self.num_choices:] if each != idx]
                    else:
                        idxs = rank[-self.num_choices + 1:]

                    # resp_distractors每一个元素都是str型的，一句话
                    # resp_distractors_bert每一个元素都是转化成bert词表对应id的List[int]
                    origin_data[key]['resp_distractors'].append([origin_data[key]['resp'][k] for k in idxs])
                    origin_data[key]['resp_distractors_bert'].append([origin_data[key]['resp_bert'][k] for k in idxs])

                with open(distractor_file, 'w', encoding="utf-8") as f:
                    json.dump(origin_data[key]['resp_distractors'], f, ensure_ascii=False, indent=4)

        logger.info(f"完成数据读取和负采样，共计用时%f s，准备构建词表" % (time.time() - begin_time))
        # 保存BERT词表
        vocab_list = [each for each in self.bert_id2word]
        valid_vocab_len = len(vocab_list)

        logger.info(f"词表数量为{len(vocab_list)}")

        # 计算每个数据集的大小
        data_size = {key: len(origin_data[key]['resp']) for key in self.key_name}
        print("数据集统计：", data_size)
        return vocab_list, valid_vocab_len, origin_data, data_size

    def get_batch(self, key, indexes):
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        res = {
            'resp': [self.data[key]['resp'][i] for i in indexes], # original response
            'can_resps': [],
            'input_ids': None,
            'input_mask': None,
            'segment_ids': None,
            'labels': None
        }
        batch_size = len(indexes)
        labels = np.zeros((batch_size * self.num_choices), dtype=int)

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []

        for iidx, idx in enumerate(indexes):
            # 表示历史输入，每一个都是List[List[int]]
            post_bert = self.data[key]['post_bert'][idx] # a list of historical utterances
            # 表示基于历史输入的回复，类型是List[int]
            resp_bert = self.data[key]['resp_bert'][idx]
            # 表示负样本，由于有多个负样本，所以类型是List[List[int]]，每一个元素是一个负样本
            resp_distractors_bert = self.data[key]['resp_distractors_bert'][idx]

            # 这里保存原始的句子，合并之后就是List[str]，每一个元素是一句话（原始语句，没有分词）
            ori_can_resps = [self.data[key]['resp'][idx]] + self.data[key]['resp_distractors'][idx]
            can_resps = []

            options = [resp_bert] + resp_distractors_bert
            # 将正样本和负样本打乱
            options = list((ii, option) for ii, option in enumerate(options))
            random.shuffle(options)
            # 保存被打乱之后原始的id索引
            # option_ids: List[int]
            # options: List[List[int]]
            option_ids = [each[0] for each in options]
            options = [each[1] for each in options]

            # 记录当前batch中的最长长度
            resp_max_length = max([len(each) for each in options])
            # 对resp_max_length增加限制条件
            if resp_max_length >= (self._max_sent_length - 3):
                # 则限制resp_max_length的长度为当前max_sent_length的一半
                resp_max_length = self._max_sent_length // 2
                options = [op[:resp_max_length] for op in options]
            # 表示历史对话的最长长度
            # 由于最终的输入形式为 [CLS] post [SEP] resp [SEP]
            # 所以这里要减去3
            post_max_length = self._max_sent_length - 3 - resp_max_length
            # 这里从后向前，留下历史对话的最长长度
            post_bert = list(chain(*post_bert))[-post_max_length:]
            for tt, (option_id, option) in enumerate(zip(option_ids, options)):
                # 这里记录当前实例的原始输入，str型
                can_resps.append(ori_can_resps[option_id])

                # 获取当前样本所需要的输入
                # input_id：表示输入单词的id
                # input_mask：表示对输入的mask向量，只保留有效的单词
                # segment_ids：表示输入的单词是历史的对话post，还是回复resp
                input_id = [self.bert_go_id] + post_bert + [self.bert_eos_id] + option + [self.bert_eos_id]
                input_mask = [1] * len(input_id)
                segment_ids = [0] * (len(post_bert) + 2) + [1] * (len(option) + 1)

                assert len(input_id) == len(segment_ids)

                padding = [0] * (self._max_sent_length - len(input_id))
                input_id = input_id + padding
                input_mask = input_mask + padding
                segment_ids = segment_ids + padding

                all_input_ids.append(input_id)
                all_input_mask.append(input_mask)
                all_segment_ids.append(segment_ids)
                # 如果option_id = 0，表示是正样本，对应的标签为1
                # 否则则是负样本，对应的标签为0
                labels[iidx * self.num_choices + tt] = 1 if option_id == 0 else 0

            res['can_resps'].append(can_resps)

        assert len(all_input_ids) == batch_size * self.num_choices

        # 将当前输入保存为一个字典
        res['input_ids'] = all_input_ids
        res['input_mask'] = all_input_mask
        res['segment_ids'] = all_segment_ids
        res['labels'] = labels

        return res

    def get_inference_metric(self, gen_key="gen"):
        '''Get metrics for inference.

        It contains:

        * :class:`.metric.BleuCorpusMetric`
        * :class:`.metric.SingleTurnDialogRecorder`

        Arguments:
            gen_key (str): The key of generated sentences in index form.
                Refer to :class:`.metric.BleuCorpusMetric` or
                :class:`.metric.SingleTurnDialogRecorder`. Default: ``gen``.
        Returns:
            A :class:`.metric.MetricChain` object.
        '''
        metric = MetricChain()
        metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key, \
            reference_allvocabs_key="resp_allvocabs"))
        metric.add_metric(SingleTurnDialogRecorder(self, gen_key=gen_key))
        metric.add_metric(SingleTurnDistinct(self, gen_key=gen_key))
        return metric


# ---------------------------- 基于知识选择的BERT检索 --------------------------------


class MyMemBERTRetrieval(BERTLanguageProcessingBase):
    def __init__(self, file_id, bert_vocab_name, do_lower_case, num_choices=10,
                 max_sent_length=192, max_know_length=100, num_turns=8,
                 ext_vocab=None, key_name=None, cpu_count=None):
        self.ext_vocab = ext_vocab or ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        # 设置句子的最大长度和知识的最大长度
        self._max_sent_length = max_sent_length
        self._max_know_length = max_know_length
        # 数据集的路径
        self._file_path = file_id
        # 负样本的数量
        self.num_choices = num_choices
        # 最大轮次数
        self.num_turns = num_turns
        super().__init__(self.ext_vocab, key_name, bert_vocab_name)

        self.tokenizer = BertTokenizer(vocab_file=bert_vocab_name, do_lower_case=do_lower_case)
        self._build_bert_vocab()

        if cpu_count is not None:
            self.cpu_count = cpu_count
        elif "CPU_COUNT" in os.environ and os.environ["CPU_COUNT"] is not None:
            self.cpu_count = int(os.environ["CPU_COUNT"])
        else:
            self.cpu_count = multiprocessing.cpu_count()

    def _load_data(self):
        r'''Loading dataset, invoked by `BERTLanguageProcessingBase.__init__`
        '''
        logger.info("开始读取和处理数据")
        begin_time = time.time()
        origin_data = {}
        vocab = {}

        def count_token(tokens):
            # 统计列表中所有词出现的次数
            for token in tokens:
                vocab[token] = vocab[token] + 1 if token in vocab else 1

        with open("../../data/resources/hit_stopwords.txt", "r", encoding="utf-8") as f:
            stop_words = set([w.strip() for w in f])

        # 这里的key_name包含["train", "dev", "test"]
        for key in self.key_name:
            corpus_resp = []     # 表示回复中文分词之后的结果，List[str]
            corpus_post = []     # 表示历史对话中文分词之后的结果，List[List[str]]
            # kg_index: List[List[int]]，表示当前回复使用的知识在kg中对应的id
            # kg: List[Tuple[Tuple[str]]]，表示当前对话所有的知识，知识都是经过中文分词的，第一个的Tuple包含(h, r, t)
            # resp: 表示当前回复对应的原文，str型
            # post_bert: 表示历史对话对应的bert词表的id，List[List[int]]
            # resp_bert: 表示当前回复对应bert词表的id，List[int]
            origin_data[key] = {'kg_index': [], 'kg': [], 'resp': [], 'post_bert': [], 'resp_bert': []}
            datas = json.load(open("%s/%s.json" % (self._file_path, key), "r", encoding="utf-8"))

            logger.info(f"当前正在处理 {key} 的数据，数据量为{len(datas)}")

            for data in datas:
                # 获取一段对话
                messages = data['messages']
                kg = []         # 用来保存当前对话的所有知识，知识经过中文分词，List[Tuple[Tuple[str]]]
                kg_index = []   # 用来保存当前对话每一个回复对应的知识索引，List[List[int]]，可能为空
                kg_dict = {}    # 用来保存知识到索引的映射
                # 对于这段对话中的每一句话
                # 每一句话都是dict型
                # 问题只包含"message"字段
                # 回答包含"attrs", "message"字段
                for message in messages:
                    kg_index.append([])
                    if 'attrs' in message:
                        for attr in message['attrs']:
                            # head实体，分词
                            h = jieba.lcut(attr['name'])
                            # 关系属性，分词
                            r = jieba.lcut(attr['attrname'])
                            # tail实体，分词
                            t = jieba.lcut(attr['attrvalue'])
                            # (h, r, t)组成一组知识
                            k = tuple((tuple(h), tuple(r), tuple(t)))
                            # 如果知识没有在字典中出现过，则添加到字典中
                            if k not in kg_dict:
                                # 当前知识的索引等于当前知识库的长度
                                kg_dict[k] = len(kg)
                                # 当前知识添加到所有知识库中
                                kg.append(k)
                            # 保存当前对话所依赖的知识索引
                            kg_index[-1].append(kg_dict[k])
                            count_token(h + r + t)  # 统计单词的频率

                # 对于当前这段话
                i = 0
                post_sent = []
                corpus_post_sent = []
                while (i + 1) < len(messages):
                    # 如果是第一句话
                    if i == 0:
                        # 将当前语句转换成bert词表中单词的id
                        tmp_sent = self.convert_tokens_to_bert_ids(self.tokenize(messages[i]['message']))
                        post_sent = [tmp_sent]
                        # 中文分词的结果
                        corpus_post_sent = [
                            [token for token in jieba.lcut(messages[i]['message']) if token not in stop_words]]

                    # 将下一句话看做是回复
                    origin_data[key]['resp'].append(messages[i + 1]['message'])
                    # 对下一句话分词
                    sentence = [token for token in jieba.lcut(messages[i + 1]['message']) if token not in stop_words]
                    # 转化成bert词表中的id
                    nxt_sent = self.convert_tokens_to_bert_ids(self.tokenize(messages[i + 1]['message']))

                    # 这里保存的是历史对话，对应单词id，List[List[int]]，第一维的每一个元素表示一个轮次
                    origin_data[key]['post_bert'].append(post_sent)
                    # 这里表示回复语句中单词对应的id，List[int]
                    origin_data[key]['resp_bert'].append(nxt_sent)
                    corpus_resp.append(sentence)           # 回复resp对应的中文分词结果，List[str]
                    corpus_post.append(corpus_post_sent)   # 历史对话post对应的中文分词结果，List[List[str]]

                    # 保存当前轮往前推算特定轮次范围内的语句
                    post_sent = (post_sent + [nxt_sent])[- self.num_turns + 1:]
                    corpus_post_sent = corpus_post_sent + [sentence]
                    origin_data[key]['kg'].append(kg)                      # 当前对话包含的所有知识
                    origin_data[key]['kg_index'].append(kg_index[i + 1])   # 当前回复所使用的知识的索引

                    i += 1

            logger.info(f"{key}文件共获取有效数据{len(corpus_post)}条")
            logger.info("采样负样本用于训练模型")

            distractor_file = os.path.join(self._file_path, '%s_distractors.json' % key)
            if os.path.exists(distractor_file):
                with open(distractor_file, "r", encoding="utf-8") as f:
                    origin_data[key]['resp_distractors'] = json.load(f)
                origin_data[key]['resp_distractors_bert'] = [
                    [self.convert_tokens_to_bert_ids(self.tokenize(sent)) for sent in distractors] for distractors in
                    origin_data[key]['resp_distractors']]

            else:
                # 基于回复构建BM25模型
                logger.info("构建BM25模型")
                bm25Model = bm25.BM25(corpus_resp)
                origin_data[key]['resp_distractors'] = []
                origin_data[key]['resp_distractors_bert'] = []

                logger.info("对有效数据中的每一个样本获取其对应的负样本")
                for idx in range(len(corpus_resp)):
                    posts = corpus_post[idx]    # 上下文
                    resp = corpus_resp[idx]

                    # 计算上下文中所有单词的频率
                    count_post_token = {}
                    for token in list(chain(*posts)):
                        if token in count_post_token:
                            count_post_token[token] += 1
                        else:
                            count_post_token[token] = 1

                    # 计算上下文中每一个单词的tfidf
                    token_tfidf = {}
                    for token in count_post_token:
                        if token in bm25Model.idf:
                            token_tfidf[token] = count_post_token[token] * bm25Model.idf[token]

                    # 选择tfidf值较大的前5个单词
                    token_tfidf = sorted(list(token_tfidf.items()), key=lambda x: (-x[1], x[0]))
                    top5 = [each[0] for each in token_tfidf[:5]]

                    # 将回复的单词和上下文中最重要的前5个单词作为查询词
                    # 寻找所有回复中和当前回复和上下文最相关的作为负样本
                    bm_scores = bm25Model.get_scores(list(set(resp + top5)))
                    bm_scores = np.array(bm_scores)
                    rank = np.argsort(bm_scores).tolist()
                    # 选择排序最高的负样本，除了本身之外
                    if idx in rank[-self.num_choices:]:
                        idxs = [each for each in rank[-self.num_choices:] if each != idx]
                    else:
                        idxs = rank[-self.num_choices + 1:]

                    origin_data[key]['resp_distractors'].append([origin_data[key]['resp'][k] for k in idxs])
                    origin_data[key]['resp_distractors_bert'].append([origin_data[key]['resp_bert'][k] for k in idxs])

                with open(distractor_file, 'w', encoding="utf-8") as f:
                    json.dump(origin_data[key]['resp_distractors'], f, ensure_ascii=False, indent=4)

        logger.info(f"完成数据读取和负采样，共计用时%f s，准备构建词表" % (time.time() - begin_time))
        # 记录单词词表
        vocab_list = [each for each in self.bert_id2word]
        valid_vocab_len = len(vocab_list)
        logger.info(f"词表数量为{len(vocab_list)}")

        # 知识库中的单词按照从高到低排序
        vocab = sorted(list(vocab.items()), key=lambda pair: (-pair[1], pair[0]))
        # 分词是索引到单词的映射和单词到索引的映射
        self.id2know_word = list(map(lambda x: x[0], vocab))
        self.know_word2id = {w: i for i, w in enumerate(self.id2know_word)}
        logger.info(f"知识库词表的长度：{len(self.id2know_word)}")

        know2id = lambda line: list(map(lambda word: self.know_word2id.get(word, self.unk_id), line))
        knows2id = lambda lines: list(map(know2id, lines))
        # 将知识库中的单词转换为知识库词表中单词对应的索引
        for key in self.key_name:
            origin_data[key]['kg'] = [list(map(knows2id, kg)) for kg in origin_data[key]['kg']]

        # 计算train, dev, test每一组数据集的大小
        data_size = {key: len(origin_data[key]['resp']) for key in self.key_name}
        print("数据集统计：", data_size)
        # origin_data就是self.data
        return vocab_list, valid_vocab_len, origin_data, data_size

    def get_batch(self, key, indexes):
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        res = {
            'resp': [self.data[key]['resp'][i] for i in indexes], # original response，str型
            'can_resps': [],
            'input_ids': None,
            'input_mask': None,
            'segment_ids': None,
            'resp_ids': None,
            'resp_mask': None,
            'labels': None
        }
        # batch的大小
        batch_size = len(indexes)
        labels = np.zeros((batch_size * self.num_choices), dtype=int)

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        resp_ids = []
        resp_mask = []
        # 当前语句正样本回复的最大长度 + 负样本回复的最大长度 - 2
        # 与预定义最大长度 - 2
        # 其中最小的那个作为当前最大长度
        single_resp_max_length = min(
            max([len(self.data[key]['resp_bert'][idx]) for idx in indexes] +
                [max([len(distractor) for distractor in self.data[key]['resp_distractors_bert'][idx]]) for idx in indexes]) - 2,
            self._max_sent_length - 2)

        for iidx, idx in enumerate(indexes):
            post_bert = self.data[key]['post_bert'][idx] # a list of historical utterances
            resp_bert = self.data[key]['resp_bert'][idx]
            resp_distractors_bert = self.data[key]['resp_distractors_bert'][idx]

            # 包含了正确回复和负样本
            ori_can_resps = [self.data[key]['resp'][idx]] + self.data[key]['resp_distractors'][idx]
            can_resps = []

            # 这里表示转化为bert id之后的列表
            options = [resp_bert] + resp_distractors_bert
            options = list((ii, option) for ii, option in enumerate(options))
            random.shuffle(options)
            option_ids = [each[0] for each in options]
            options = [each[1] for each in options]

            resp_max_length = max([len(each) for each in options])
            # 对resp_max_length增加限制条件
            if resp_max_length >= (self._max_sent_length - 3):
                # 则限制resp_max_length的长度为当前max_sent_length的一半
                resp_max_length = self._max_sent_length // 2
                options = [op[:resp_max_length] for op in options]

            # 这里减去3因为一个[CLS]和两个[SEP]
            post_max_length = self._max_sent_length - 3 - resp_max_length
            # 选择历史回复中后面的靠后的单词
            post_bert = list(chain(*post_bert))[-post_max_length:]

            for tt, (option_id, option) in enumerate(zip(option_ids, options)):
                can_resps.append(ori_can_resps[option_id])

                resp_id = [self.bert_go_id] + option[:single_resp_max_length] + [self.bert_eos_id]
                resp_msk = [1] * len(resp_id)
                padding = [0] * (single_resp_max_length + 2 - len(resp_id))
                resp_ids.append(resp_id + padding)
                resp_mask.append(resp_msk + padding)

                input_id = [self.bert_go_id] + post_bert + [self.bert_eos_id] + option + [self.bert_eos_id]
                input_mask = [1] * len(input_id)
                segment_ids = [0] * (len(post_bert) + 2) + [1] * (len(option) + 1)

                assert len(input_id) == len(segment_ids)

                padding = [0] * (self._max_sent_length - len(input_id))
                input_id = input_id + padding
                input_mask = input_mask + padding
                segment_ids = segment_ids + padding

                all_input_ids.append(input_id)
                all_input_mask.append(input_mask)
                all_segment_ids.append(segment_ids)
                labels[iidx * self.num_choices + tt] = 1 if option_id == 0 else 0

            res['can_resps'].append(can_resps)

        assert len(all_input_ids) == batch_size * self.num_choices

        res['input_ids'] = all_input_ids
        res['input_mask'] = all_input_mask
        res['segment_ids'] = all_segment_ids
        res['labels'] = labels
        res['resp_ids'] = resp_ids
        res['resp_mask'] = resp_mask

        # 计算每一句话中知识（kg）最多的数量是多少
        max_kg_num = max([len(self.data[key]['kg'][idx]) for idx in indexes])
        # batch_size * max_kg_num
        res["kg_h_length"] = np.zeros((batch_size, max_kg_num), dtype=int)
        res["kg_hr_length"] = np.zeros((batch_size, max_kg_num), dtype=int)
        res['kg_hrt_length'] = np.zeros((batch_size, max_kg_num), dtype=int)
        # 每一个知识中h, hr, hrt和预定义最大值相比
        # 取其中较小值
        for i, idx in enumerate(indexes):
            kg_h_length = [min(self._max_know_length, len(sent[0])) for sent in self.data[key]['kg'][idx]]
            res["kg_h_length"][i, :len(kg_h_length)] = kg_h_length
            kg_hr_length = [min(self._max_know_length, len(sent[0]) + len(sent[1])) for sent in self.data[key]['kg'][idx]]
            res["kg_hr_length"][i, :len(kg_hr_length)] = kg_hr_length
            kg_hrt_length = [min(self._max_know_length, len(sent[0]) + len(sent[1]) + len(sent[2])) for sent in
                             self.data[key]['kg'][idx]]
            res["kg_hrt_length"][i, :len(kg_hrt_length)] = kg_hrt_length

        # batch_size * max_kg_num * max_kg_hrt_length
        res['kg'] = np.zeros((batch_size, max_kg_num, np.max(res["kg_hrt_length"])), dtype=int)
        for i, idx in enumerate(indexes):
            for j, tri in enumerate(self.data[key]['kg'][idx]):
                # 获取知识中单词对应的索引
                sent = (tri[0] + tri[1] + tri[2])[:self._max_know_length]
                res['kg'][i, j, :len(sent)] = sent

        # 将每一个样例中使用的知识对应的位置设为1
        res['kg_index'] = np.zeros((batch_size, max_kg_num), dtype=float)
        for i, idx in enumerate(indexes):
            for kgid in self.data[key]['kg_index'][idx]:
                res['kg_index'][i, kgid] = 1

        # 扩充之后维度为 [batch_size * num_choices, * ....]
        for k in ['kg_h_length', 'kg_hr_length', 'kg_hrt_length', 'kg', 'kg_index']:
            res[k] = res[k].repeat(self.num_choices, 0)

        return res

    def get_inference_metric(self, gen_key="gen"):
        '''Get metrics for inference.

        It contains:

        * :class:`.metric.BleuCorpusMetric`
        * :class:`.metric.SingleTurnDialogRecorder`

        Arguments:
            gen_key (str): The key of generated sentences in index form.
                Refer to :class:`.metric.BleuCorpusMetric` or
                :class:`.metric.SingleTurnDialogRecorder`. Default: ``gen``.
        Returns:
            A :class:`.metric.MetricChain` object.
        '''
        metric = MetricChain()
        metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key, \
            reference_allvocabs_key="resp_allvocabs"))
        metric.add_metric(SingleTurnDialogRecorder(self, gen_key=gen_key))
        metric.add_metric(SingleTurnDistinct(self, gen_key=gen_key))
        return metric


# ---------------------------- 基于GPT的生成模型 --------------------------------


class GPTGen(BERTLanguageProcessingBase):
    def __init__(self, file_id: str,
                 vocab_name: str,
                 do_lower_case: bool = True,
                 max_sent_length: int = 256,
                 num_turns: int = 8,
                 is_relative: bool = True,
                 ext_vocab: Optional[List[str]] = None,
                 key_name: Optional[List[str]] = None,
                 cpu_count: Optional[int] = None):
        """
        用于GPT模型的数据读取类
        由于GPT的token_type_ids是基于词向量的embedding进行映射的，所以可以直接使用到vocab_size的大小
        file_id: str，表示数据所在的文件夹路径
        key_name: List[str]，默认为["train", "dev", "test"]
        """
        self.speakers = ["[speaker1]", "[speaker2]"]
        self.ext_vocab = ext_vocab or ["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + self.speakers
        self._max_sent_length = max_sent_length
        self._file_path = file_id
        self.num_turns = num_turns
        self.is_relative = is_relative

        self.tokenizer = BertTokenizer(vocab_file=vocab_name, do_lower_case=do_lower_case)
        self._build_bert_vocab()
        self.speakers_id = self.convert_tokens_to_bert_ids(self.speakers)

        super().__init__(self.ext_vocab, key_name, vocab_name)

        if cpu_count is not None:
            self.cpu_count = cpu_count
        elif "CPU_COUNT" in os.environ and os.environ["CPU_COUNT"] is not None:
            self.cpu_count = int(os.environ["CPU_COUNT"])
        else:
            self.cpu_count = multiprocessing.cpu_count()

    def _load_data(self):
        """
        Loaing dataset
        """
        logger.info("开始读取和处理数据")
        begin_time = time.time()
        origin_data = {}

        for key in self.key_name:
            origin_data[key] = {"posts": [], "prev_posts": [], "responses": [], "origin_responses": []}
            datas = json.load(open(f"{self._file_path}/{key}.json", "r", encoding="utf-8"))
            for data in datas:
                messages = data["messages"]
                turns = []  # 保存当前一段所有对话
                for message in messages:
                    # 对当前语句进行分词
                    sent = self.tokenize(message["message"])
                    turns.append(sent)

                # 得到一段对话的所有语句之后，就可以得到该轮对话的所有训练样本
                for i in range(len(turns) - 1):
                    # 保存一个样本以及对应的回复
                    # 这里获取回复对应的历史对话
                    posts = []
                    cur_speaker = self.speakers_id[0]
                    for j in range(max(0, (i+1)-(self.num_turns-1)), i+1):
                        cur_speaker = self.speakers_id[0] if j % 2 == 0 else self.speakers_id[1]
                        posts.append([cur_speaker] + self.convert_tokens_to_bert_ids(turns[j]))
                    prev_post = posts[-1]
                    # 获取回复对应的说话人id
                    next_speaker = self.speakers_id[0] if cur_speaker == self.speakers_id[1] else self.speakers_id[1]
                    response = [next_speaker] + self.convert_tokens_to_bert_ids(turns[i+1])
                    origin_response = turns[i+1]

                    origin_data[key]["posts"].append(posts)
                    origin_data[key]["prev_posts"].append(prev_post)
                    origin_data[key]["responses"].append(response)
                    origin_data[key]["origin_responses"].append(origin_response)

        logger.info(f"数据读取用时 {(time.time() - begin_time) * 1000} ms")
        # 保存bert词表
        vocab_list = [each for each in self.bert_id2word]
        valid_vocab_len = len(vocab_list)
        logger.info(f"词表的数量为：{valid_vocab_len}")

        # 计算每一个数据集的大小
        data_size = {key: len(origin_data[key]['responses']) for key in self.key_name}
        print("数据集统计：", data_size)
        return vocab_list, valid_vocab_len, origin_data, data_size

    def get_batch(self, key, indexes):
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        res = {
            "resp": [self.data[key]["origin_responses"][i] for i in indexes],  # 表示回复的原文本
            "resp_lens": [len(self.data[key]["origin_responses"][i]) for i in indexes],  # 表示回复原文本的长度
            "posts_lens": None,
            "input_ids": None,
            "input_mask": None,
            "token_type_ids": None,
            "turn_ids": None,
            "lm_labels": None
        }

        all_input_ids, all_input_mask, all_token_type_ids, all_turn_ids, all_lm_labels = [], [], [], [], []
        posts_lens = []

        for iidx, idx in enumerate(indexes):
            # 获取上下文的语句和当前回复语句
            # 这里的语句都已经转化为了id的形式
            # 并且每个语句的第一个token都表示对话人的标识
            posts = self.data[key]["posts"][idx]
            response = self.data[key]["responses"][idx]

            # 计算当前response的长度
            resp_len = len(response)
            # 如果当前的回复长度大于允许的最大长度
            # 由于头尾有[CLS]和[SEP]
            if resp_len > self._max_sent_length - 2:
                response = response[:self._max_sent_length-2]
                allow_max_length = 0
            else:
                allow_max_length = self._max_sent_length - 2 - resp_len
            # 对posts的长度进行裁剪
            posts = self.trim_posts(posts, allow_max_length)

            # 获取token_type_ids
            resp_token_type_ids = self.get_token_type_ids(response)
            posts_token_type_ids = []
            for post in posts:
                posts_token_type_ids.append(self.get_token_type_ids(post))

            # 获取turn_ids
            if self.is_relative:
                posts_turns, resp_turns = self.get_relative_turns(posts, response)
            else:
                posts_turns, resp_turns = self.get_absolute_turns(posts, response)


            # 首先将上面所有的post转化为列表形式
            posts_input = list(chain(*posts))
            posts_token_type_ids = list(chain(*posts_token_type_ids))
            posts_turns = list(chain(*posts_turns))
            if key != "test":
                # 转化为GPT2的输入
                posts_len = len(posts_input) + 1   # 包含go_id
                input_ids = [self.bert_go_id] + posts_input + response + [self.bert_eos_id]
                token_type_ids = [self.bert_go_id] + posts_token_type_ids + resp_token_type_ids + [self.bert_eos_id]
                turn_ids = [posts_turns[0]] + posts_turns + resp_turns + [resp_turns[-1]]
                input_mask = [1] * len(input_ids)

                # 获取对应的单词标签
                # 在GPT内部计算loss的时候有shift的操作，所以这里只需要逐一对齐即可
                lm_labels = [-1] * posts_len + [-1] + response[1:] + [self.bert_eos_id]

                assert len(input_ids) == len(turn_ids)
                assert len(input_ids) == len(lm_labels)

                padding = [0] * (self._max_sent_length - len(input_ids))
                input_ids = input_ids + padding
                token_type_ids = token_type_ids + padding
                turn_ids = turn_ids + padding
                input_mask = input_mask + padding
                lm_labels = lm_labels + [-1] * len(padding)


                all_input_ids.append(input_ids)
                all_token_type_ids.append(token_type_ids)
                all_turn_ids.append(turn_ids)
                all_input_mask.append(input_mask)
                all_lm_labels.append(lm_labels)
                posts_lens.append(posts_len)
            else:
                # 预测时的输入是不包含response的
                # 对预测输入的封装不进行padding
                posts_len = len(posts_input) + 1
                input_ids = [self.bert_go_id] + posts_input + [response[0]]  # 这里相当于是用于生成response的起始符
                token_type_ids = [self.bert_go_id] + posts_token_type_ids + [resp_token_type_ids[0]]
                turn_ids = [posts_turns[0]] + posts_turns + [resp_turns[0]]

                assert len(input_ids) == len(turn_ids)

                all_input_ids.append(input_ids)
                all_token_type_ids.append(token_type_ids)
                all_turn_ids.append(turn_ids)
                posts_lens.append(posts_len)

        # 将当前输入保存到词典中
        res["input_ids"] = all_input_ids
        res["input_mask"] = all_input_mask
        res["token_type_ids"] = all_token_type_ids
        res["turn_ids"] = all_turn_ids
        res["lm_labels"] = all_lm_labels
        res["posts_lens"] = posts_lens
        return res

    def get_token_type_ids(self, sent: List[int]):
        token_type_ids = []
        if len(sent) > 0:
            token_type_ids = [sent[0]] * len(sent)
        return token_type_ids

    def trim_posts(self, posts: List[List[int]], allow_max_length: int):
        """
        根据最大长度对posts进行裁剪，从后往前
        """
        result_posts = []
        reverse_posts = posts[::-1]
        total_length = 0
        for i, post in enumerate(reverse_posts):
            cur_length = len(post)
            if total_length + cur_length > allow_max_length:
                return result_posts
            else:
                total_length += cur_length
                result_posts = [post] + result_posts
        return result_posts

    def get_relative_turns(self, posts: List[List[int]], response: List[int]):
        """
        获取相对的posts_turns和response_turns
        """
        response_turns = [0] * len(response)
        posts_turns = []
        reverse_posts = posts[::-1]
        for i, post in enumerate(reverse_posts):
            cur_turns = [i+1] * len(post)
            posts_turns = [cur_turns] + posts_turns
        return posts_turns, response_turns

    def get_absolute_turns(self, posts: List[List[int]], response: List[int]):
        """
        获取绝对的posts_turns和response_turns
        """
        cur_turn = 0
        posts_turns = []
        for i, post in enumerate(posts):
            cur_turn = i
            cur_turns = [cur_turn] * len(post)
            posts_turns.append(cur_turns)
        response_turns = [cur_turn + 1] * len(response)
        return posts_turns, response_turns

    def get_teacher_forcing_metric(self,
                                   gen_log_prob_key: str = "gen_log_prob",
                                   invalid_vocab: bool = False):
        metric = MetricChain()
        metric.add_metric(
            PerplexityMetric(self, reference_allvocabs_key="resp_allvocabs",
                             reference_len_key="resp_length",
                             gen_log_prob_key=gen_log_prob_key,
                             invalid_vocab=invalid_vocab)
        )
        return metric

    def get_inference_metric(self, gen_key: str = "gen"):
        metric = MetricChain()
        metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key, reference_allvocabs_key="resp_allvocabs"))
        metric.add_metric(SingleTurnResponseRecorder(self, gen_key=gen_key))
        metric.add_metric(SingleTurnDistinct(self, gen_key=gen_key))
        return metric


# ---------------------------- 基于知识选择的GPT生成 --------------------------------


class GPTGenKA(GPTGen):
    """基于知识注意力机制的GPT生成模型"""
    def __init__(self, file_id: str,
                 vocab_name: str,
                 do_lower_case: bool = True,
                 max_sent_length: int = 256,
                 max_know_length: int = 128,
                 num_turns: int = 8,
                 is_relative: bool = True,
                 ext_vocab: Optional[List[str]] = None,
                 key_name: Optional[List[str]] = None,
                 cpu_count: Optional[int] = None):
        """
        用于GPT模型的数据读取类
        由于GPT的token_type_ids是基于词向量的embedding进行映射的，所以可以直接使用到vocab_size的大小
        file_id: str，表示数据所在的文件夹路径
        key_name: List[str]，默认为["train", "dev", "test"]
        """
        super().__init__(file_id, vocab_name, do_lower_case,
                         max_sent_length, num_turns, is_relative,
                         ext_vocab, key_name, cpu_count)
        self._max_know_length = max_know_length

    def _load_data(self):
        logger.info("开始读取和处理数据")
        begin_time = time.time()
        origin_data = {}
        vocab = {}

        def count_token(tokens):
            # 统计列表中所有词出现的次数
            for token in tokens:
                vocab[token] = vocab[token] + 1 if token in vocab else 1

        for key in self.key_name:
            origin_data[key] = {"posts": [], "prev_posts": [], "responses": [], "origin_responses": [],
                                "kg_index": [], "kg": []}
            datas = json.load(open("%s/%s.json" % (self._file_path, key), "r", encoding="utf-8"))

            logger.info(f"当前正在处理 {key} 的数据，数据量为{len(datas)}")
            # 对于每一段对话
            for data in datas:
                # 获取当前对话的内容
                messages = data["messages"]
                kg = []       # 保存当前对话的所有知识
                kg_index = []
                kg_dict = {}
                turns = []    # 保存当前对话的所有内容
                for message in messages:
                    kg_index.append([])
                    # 对当前语句进行分词
                    sent = self.tokenize(message["message"])
                    turns.append(sent)
                    if "attrs" in message:
                        for attr in message["attrs"]:
                            h = jieba.lcut(attr["name"])
                            r = jieba.lcut(attr["attrname"])
                            t = jieba.lcut(attr["attrvalue"])
                            # 一个三元组构成知识
                            k = tuple((tuple(h), tuple(r), tuple(t)))
                            # 如果当前的知识没有在字典中出现过，则添加到字典中
                            if k not in kg_dict:
                                kg_dict[k] = len(kg)
                                kg.append(k)
                            kg_index[-1].append(kg_dict[k])
                            count_token(h + r + t)

                # 得到一段对话的所有语句之后，就可以得到该轮对话的所有训练样本
                for i in range(len(turns) - 1):
                    posts = []
                    cur_speaker = self.speakers_id[0]
                    for j in range(max(0, (i+1)-(self.num_turns-1)), i+1):
                        cur_speaker = self.speakers_id[0] if j % 2 == 0 else self.speakers_id[1]
                        posts.append([cur_speaker] + self.convert_tokens_to_bert_ids(turns[j]))
                    prev_posts = posts[-1]
                    # 获取回复对应的对话人id
                    next_speaker = self.speakers_id[0] if cur_speaker == self.speakers_id[1] else self.speakers_id[1]
                    response = [next_speaker] + self.convert_tokens_to_bert_ids(turns[i+1])
                    origin_response = turns[i+1]

                    origin_data[key]["posts"].append(posts)
                    origin_data[key]["prev_posts"].append(prev_posts)
                    origin_data[key]["responses"].append(response)
                    origin_data[key]["origin_responses"].append(origin_response)
                    # 保存当前对话中所有的知识
                    origin_data[key]["kg"].append(kg)
                    origin_data[key]["kg_index"].append(kg_index[i+1])  # 当前回复所使用的知识索引

        logger.info(f"完成数据读取，共计用时{time.time()-begin_time} s，准备构建词表")
        # 这里保存的是bert单词词表
        vocab_list = [each for each in self.bert_id2word]
        valid_vocab_len = len(vocab_list)
        logger.info(f"词表数量为 {len(vocab_list)}")

        # 知识库中的单词按照从高到低排序
        vocab = sorted(list(vocab.items()), key=lambda pair: (-pair[1], pair[0]))
        self.id2know_word = list(map(lambda x: x[0], vocab))
        self.know_word2id = {w:i for i, w in enumerate(self.id2know_word)}
        logger.info(f"知识库词表的长度：{len(self.id2know_word)}")

        know2id = lambda line: list(map(lambda word: self.know_word2id.get(word, self.unk_id), line))
        knows2id = lambda lines: list(map(know2id, lines))
        # 将知识库中的单词转换为知识库词表中单词对应的索引
        for key in self.key_name:
            origin_data[key]["kg"] = [list(map(knows2id, kg)) for kg in origin_data[key]["kg"]]

        # 计算train, dev, test每一组数据集的大小
        data_size = {key: len(origin_data[key]["responses"]) for key in self.key_name}
        print("数据集统计：", data_size)
        return vocab_list, valid_vocab_len, origin_data, data_size


    def get_batch(self, key, indexes):
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)

        batch_size = len(indexes)
        res = {
            "resp": [self.data[key]["origin_responses"][i] for i in indexes],  # 表示回复的原文本
            "resp_lens": [len(self.data[key]["origin_responses"][i]) for i in indexes],  # 表示回复原文本的长度
            "posts_lens": None,
            "input_ids": None,
            "input_mask": None,
            "token_type_ids": None,
            "turn_ids": None,
            "lm_labels": None
        }

        all_input_ids, all_input_mask, all_token_type_ids, all_turn_ids, all_lm_labels = [], [], [], [], []
        posts_lens = []

        for iidx, idx in enumerate(indexes):
            # 获取上下文的语句和当前回复语句
            # 这里的语句都已经转化为了id的形式
            # 并且每个语句的第一个token都表示对话人的标识
            posts = self.data[key]["posts"][idx]
            response = self.data[key]["responses"][idx]

            # 计算当前response的长度
            resp_len = len(response)
            # 如果当前的回复长度大于允许的最大长度
            # 由于头尾有[CLS]和[SEP]
            if resp_len > self._max_sent_length - 2:
                response = response[:self._max_sent_length-2]
                allow_max_length = 0
            else:
                allow_max_length = self._max_sent_length - 2 - resp_len
            # 对posts的长度进行裁剪
            posts = self.trim_posts(posts, allow_max_length)

            # 获取token_type_ids
            resp_token_type_ids = self.get_token_type_ids(response)
            posts_token_type_ids = []
            for post in posts:
                posts_token_type_ids.append(self.get_token_type_ids(post))

            # 获取turn_ids
            if self.is_relative:
                posts_turns, resp_turns = self.get_relative_turns(posts, response)
            else:
                posts_turns, resp_turns = self.get_absolute_turns(posts, response)


            # 首先将上面所有的post转化为列表形式
            posts_input = list(chain(*posts))
            posts_token_type_ids = list(chain(*posts_token_type_ids))
            posts_turns = list(chain(*posts_turns))
            if key != "test":
                # 转化为GPT2的输入
                posts_len = len(posts_input) + 1   # 包含go_id
                input_ids = [self.bert_go_id] + posts_input + response + [self.bert_eos_id]
                token_type_ids = [self.bert_go_id] + posts_token_type_ids + resp_token_type_ids + [self.bert_eos_id]
                turn_ids = [posts_turns[0]] + posts_turns + resp_turns + [resp_turns[-1]]
                input_mask = [1] * len(input_ids)

                # 获取对应的单词标签
                # 在GPT内部计算loss的时候有shift的操作，所以这里只需要逐一对齐即可
                lm_labels = [-1] * posts_len + [-1] + response[1:] + [self.bert_eos_id]

                assert len(input_ids) == len(turn_ids)
                assert len(input_ids) == len(lm_labels)

                padding = [0] * (self._max_sent_length - len(input_ids))
                input_ids = input_ids + padding
                token_type_ids = token_type_ids + padding
                turn_ids = turn_ids + padding
                input_mask = input_mask + padding
                lm_labels = lm_labels + [-1] * len(padding)


                all_input_ids.append(input_ids)
                all_token_type_ids.append(token_type_ids)
                all_turn_ids.append(turn_ids)
                all_input_mask.append(input_mask)
                all_lm_labels.append(lm_labels)
                posts_lens.append(posts_len)
            else:
                # 预测时的输入是不包含response的
                # 对预测输入的封装不进行padding
                posts_len = len(posts_input) + 1
                input_ids = [self.bert_go_id] + posts_input + [response[0]]  # 这里相当于是用于生成response的起始符
                token_type_ids = [self.bert_go_id] + posts_token_type_ids + [resp_token_type_ids[0]]
                turn_ids = [posts_turns[0]] + posts_turns + [resp_turns[0]]

                assert len(input_ids) == len(turn_ids)

                all_input_ids.append(input_ids)
                all_token_type_ids.append(token_type_ids)
                all_turn_ids.append(turn_ids)
                posts_lens.append(posts_len)

        # 将当前输入保存到词典中
        res["input_ids"] = all_input_ids
        res["input_mask"] = all_input_mask
        res["token_type_ids"] = all_token_type_ids
        res["turn_ids"] = all_turn_ids
        res["lm_labels"] = all_lm_labels
        res["posts_lens"] = posts_lens

        # 计算当前batch中每一句话中kg最多的数量
        max_kg_num = max([len(self.data[key]['kg'][idx]) for idx in indexes])
        # batch_size * mag_kg_num
        res["kg_h_length"] = np.zeros((batch_size, max_kg_num), dtype=int)
        res["kg_hr_length"] = np.zeros((batch_size, max_kg_num), dtype=int)
        res["kg_hrt_length"] = np.zeros((batch_size, max_kg_num), dtype=int)
        # 每一组知识中的h, hr, hrt和预定义的最大值相比
        # 取其中较小值
        for i, idx in enumerate(indexes):
            kg_h_length = [min(self._max_know_length, len(sent[0])) for sent in self.data[key]["kg"][idx]]
            res["kg_h_length"][i, :len(kg_h_length)] = kg_h_length
            kg_hr_length = [min(self._max_know_length, len(sent[0])+len(sent[1])) for sent in self.data[key]["kg"][idx]]
            res["kg_hr_length"][i, :len(kg_hr_length)] = kg_hr_length
            kg_hrt_length = [min(self._max_know_length, len(sent[0])+len(sent[1])+len(sent[2]))
                             for sent in self.data[key]["kg"][idx]]
            res["kg_hrt_length"][i, :len(kg_hrt_length)] = kg_hrt_length

        # batch_size * max_kg_num * max_kg_hrt_length
        res["kg"] = np.zeros((batch_size, max_kg_num, np.max(res["kg_hrt_length"])), dtype=int)
        for i, idx in enumerate(indexes):
            for j, tri in enumerate(self.data[key]["kg"][idx]):
                # 获取知识中单词对应的索引
                sent = (tri[0] + tri[1] + tri[2])[self._max_know_length]
                res["kg"][i, j, :len(sent)] = sent

        # 将每一个样例中使用的知识对应位置设为1
        res["kg_index"] = np.zeros((batch_size, max_kg_num), dtype=float)
        for i, idx in enumerate(indexes):
            for kgid in self.data[key]["kg_index"][idx]:
                res["kg_index"][i, kgid] = 1

        return res


# ---------------------------- 基于知识树的GPT生成 --------------------------------


class GPTGenKT(GPTGen):
    def __init__(self, file_id: str,
                 vocab_name: str,
                 do_lower_case: bool = True,
                 max_sent_length: int = 256,
                 max_know_length: int = 128,
                 num_turns: int = 8,
                 is_relative: bool = True,
                 ext_vocab: Optional[List[str]] = None,
                 key_name: Optional[List[str]] = None,
                 cpu_count: Optional[int] = None):
        """
        用于GPT模型的数据读取类
        由于GPT的token_type_ids是基于词向量的embedding进行映射的，所以可以直接使用到vocab_size的大小
        file_id: str，表示数据所在的文件夹路径
        key_name: List[str]，默认为["train", "dev", "test"]
        """
        super().__init__(file_id, vocab_name, do_lower_case,
                         max_sent_length, num_turns, is_relative,
                         ext_vocab, key_name, cpu_count)
        self._max_know_length = max_know_length

    def _load_data(self):
        logger.info("开始读取和处理数据")
        begin_time = time.time()
        origin_data = {}

        for key in self.key_name:
            origin_data[key] = {"posts": [], "prev_posts": [], "responses": [], "origin_responses": [],
                                "kg_index": [], "kg": [], "know_pos": [], "know_rt": []}
            datas = json.load(open("%s/%s.json" % (self._file_path, key), "r", encoding="utf-8"))

            logger.info(f"当前正在处理 {key} 的数据，数据量为{len(datas)}")
            # 对于每一段对话
            for data in datas:
                # 获取当前对话的内容
                messages = data["messages"]
                kg = []         # 表示当前对话中所有知识
                kg_dict = {}    # 表示当前对话中所有知识到序号的映射
                kg_h_rt = {}    # 保存head实体到relation和tail之间的映射，键h是分词之后的list，值的每个元素是[r_list, t_list]
                kg_hlist_index = []   # 按照顺序保存hlist
                kg_index = []    # 表示当前对话使用的kg三元组，保存的是其在知识列表kg中的索引
                turns = []
                for message in messages:
                    kg_index.append([])
                    # 对当前语句进行分词
                    sent = self.tokenize(message["message"])
                    turns.append(sent)
                    if "attrs" in message:
                        for attr in message["attrs"]:
                            h = attr["name"]        # 头实体
                            r = attr["attrname"]    # 关系
                            t = attr["attrvalue"]   # 尾实体
                            # 一组三元组组成知识
                            k = tuple([h, r, t])
                            # 首先保证之前没有出现过
                            if k not in kg_dict:
                                kg_dict[k] = len(kg)
                                kg.append(k)
                                # 对h, r, t进行分词
                                h_list = self.tokenize(h)
                                r_list = self.convert_tokens_to_bert_ids(self.tokenize(r))
                                t_list = self.convert_tokens_to_bert_ids(self.tokenize(t))
                                # 将r和t添加到对应的h中
                                if h_list not in kg_h_rt:
                                    kg_hlist_index[h_list] = len(kg_h_rt)
                                    kg_h_rt[h_list] = [[r_list, t_list]]   # 这里已经转化成了id
                                else:
                                    kg_h_rt[h_list].append([[r_list, t_list]])
                            kg_index[-1].append(kg_dict[k])
                # 对于turns中的每一句话，记录其中包含的知识r和t
                # 这里sent是分词之后的list
                include_know = []   # List[List[List[r_list, t_list]]]，这里r_list和t_list都是List[int]
                know_h_index = []   # 表示该句中出现的知识对应的索引
                pos_pairs = []      # List[List[int]]
                for sent in turns:
                    single_turn_khi = []
                    single_turn_ik = []
                    single_turn_pp = []
                    # 遍历所有知识的head实体
                    for h_list in kg_h_rt.keys():
                        occur_pos = find_lcs(sent, h_list)    # 找到sent中每一个token在h_list中出现的位置
                        pos_pair = find_substring_pos_pair(sent, occur_pos, tokenizer=self.tokenizer)
                        if pos_pair[0] != -1:
                            single_turn_khi.append(kg_hlist_index[h_list])
                            single_turn_pp.append(pos_pair[0])
                            # 这里的kg_h_rt[h_list]是由一系列[r_list, t_list]组成的列表
                            single_turn_ik.append(kg_h_rt[h_list])
                    include_know.append(single_turn_ik)
                    pos_pairs.append(single_turn_pp)
                    know_h_index.append(single_turn_khi)  # 表示当前居中对应知识head实体的索引


                assert len(include_know) == len(turns)
                assert len(pos_pairs) == len(turns)

                # 得到一段对话的所有语句之后，就可以得到该轮对话中的所有训练样本
                for i in range(len(turns) - 1):
                    posts = []       # 保存历史对话
                    know_pos = []    # 表示当前轮使用的知识插入在哪个位置之后
                    know_rt = []     # 表示当前轮要插入的知识分词之后的结果
                    occur_h_index = set([])

                    last_speaker = self.speakers_id[0]
                    # 从后向前遍历
                    for j in range(i, max(0, (i+1)-(self.num_turns-1))-1, -1):
                        cur_speaker = self.speakers_id[0] if j % 2 == 0 else self.speakers_id[1]
                        if j == i:
                            last_speaker = cur_speaker
                        posts = [[cur_speaker] + self.convert_tokens_to_bert_ids(turns[j])] + posts
                        # 删除当前语句中哪些在后面语句中出现过的知识
                        cur_know_pos = []
                        cur_know_rt = []
                        for st_khi, st_pp, st_ik in zip(know_h_index[j], pos_pairs[j], include_know[j]):
                            # 对于出现过的知识head实体的索引
                            for khii, khi in enumerate(st_khi):
                                if khi not in occur_h_index:
                                    occur_h_index.add(khi)
                                    cur_know_pos.append(st_pp[khii]+1)   # 这里因为前面加上了对话人speaker_id，所以+1
                                    cur_know_rt.append(st_ik[khii])
                        know_pos = [cur_know_pos] + know_pos
                        know_rt = [cur_know_rt] + know_rt

                    prev_posts = posts[-1]
                    # 获取回复对应的对话人id
                    next_speaker = self.speakers_id[0] if last_speaker == self.speakers_id[1] else self.speakers_id[1]
                    response = [next_speaker] + self.convert_tokens_to_bert_ids(turns[i+1])
                    origin_response = turns[i+1]

                    # 将需要使用的知识添加到字典中
                    origin_data[key]["posts"].append(posts)
                    origin_data[key]["prev_posts"].append(prev_posts)
                    origin_data[key]["responses"].append(response)
                    origin_data[key]["origin_responses"].append(origin_response)
                    # 保存当前对话中所有的知识
                    origin_data[key]["kg"].append(kg)
                    origin_data[key]["kg_index"].append(kg_index[i+1])  # 当前回复所使用的知识索引

                    # 记录知识在每一轮中出现的位置和包含的rt分词转化为索引的结果
                    origin_data[key]["know_pos"].append(know_pos)
                    origin_data[key]["know_rt"].append(know_rt)


        logger.info(f"完成数据读取和处理，共计用时{time.time()-begin_time} s")

        # 保存bert词表
        vocab_list = [each for each in self.bert_id2word]
        valid_vocab_len = len(vocab_list)
        logger.info(f"词表的数量为：{valid_vocab_len}")

        # 计算train, dev, test每一组数据集的大小
        data_size = {key: len(origin_data[key]["responses"]) for key in self.key_name}
        print("数据集统计：", data_size)
        return vocab_list, valid_vocab_len, origin_data, data_size

    def get_batch(self, key, indexes):
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        res = {
            "resp": [self.data[key]["origin_responses"][i] for i in indexes],  # 表示回复的原文本
            "resp_lens": [len(self.data[key]["origin_responses"][i]) for i in indexes],  # 表示回复原文本的长度
            "posts_lens": None,
            "input_ids": None,
            "input_mask": None,
            "token_type_ids": None,
            "turn_ids": None,
            "source_ids": None,
            "position_ids": None,
            "lm_labels": None
        }

        all_input_ids, all_input_mask, all_token_type_ids, all_turn_ids, all_lm_labels = [], [], [], [], []
        all_source_ids, all_position_ids = [], []
        posts_lens = []

        for iidx, idx in enumerate(indexes):
            # 获取上下文的语句和当前回复语句
            # 这里的语句都已经转化为了id的形式
            # 并且每个语句的第一个token都表示对话人的标识
            posts = self.data[key]["posts"][idx]
            response = self.data[key]["responses"][idx]
            know_pos = self.data[key]["know_pos"][idx]    # 表示每个知识要插入的位置
            know_rt = self.data[key]["know_rt"][idx]      # 表示每个位置插入的内容

            # 计算当前response的长度
            resp_len = len(response)
            # 如果当前的回复长度大于允许的最大长度
            # 由于头尾有[CLS]和[SEP]
            if resp_len > self._max_sent_length - 2:
                response = response[:self._max_sent_length - 2]
                allow_max_length = 0
            else:
                allow_max_length = self._max_sent_length - 2 - resp_len
            # 对posts的长度进行裁剪
            posts = self.trim_posts(posts, allow_max_length)

            # 表示当前上下文轮次的上次
            turn_len = len(posts)
            if turn_len > 0:
                know_pos = know_pos[-turn_len:]
                know_rt = know_rt[-turn_len:]

            # 获取token_type_ids
            resp_token_type_ids = self.get_token_type_ids(response)
            posts_token_type_ids = []
            for post in posts:
                posts_token_type_ids.append(self.get_token_type_ids(post))

            # 获取turn_ids
            if self.is_relative:
                posts_turns, resp_turns = self.get_relative_turns(posts, response)
            else:
                posts_turns, resp_turns = self.get_absolute_turns(posts, response)

            # 处理知识并获取对应的id
            (know_input_ids, know_turn_ids, know_token_type_ids,
             know_position_ids, know_src_ids,
             posts_position_ids, posts_src_ids) = self.merge_know_to_posts(know_pos,
                                                                           know_rt,
                                                                           posts,
                                                                           posts_turns,
                                                                           posts_token_type_ids)
            # 首先将posts嵌套列表全部转化为列表的形式
            posts_input_ids = list(chain(*posts))
            posts_turn_ids = list(chain(*posts_turns))
            posts_token_type_ids = list(chain(*posts_token_type_ids))
            posts_position_ids = list(chain(*posts_position_ids))
            posts_src_ids = list(chain(*posts_src_ids))

            # 将知识的输入并入到posts的前面
            history_input_ids = know_input_ids + posts_input_ids
            history_turn_ids = know_turn_ids + posts_turn_ids
            history_token_type_ids = know_token_type_ids + posts_token_type_ids
            history_position_ids = know_position_ids + posts_position_ids
            # 考虑到在history前面需要加上go_id，所以history中的所有id都要+1
            history_position_ids = [pid+1 for pid in history_position_ids]
            history_src_ids = know_src_ids + posts_src_ids
            if key != "test":
                # 转化为GPT2的输入
                posts_len = len(history_input_ids) + 1  # 包含go_id, knowledge和posts
                input_ids = [self.bert_go_id] + history_input_ids + response + [self.bert_eos_id]
                token_type_ids = [self.bert_go_id] + history_token_type_ids + resp_token_type_ids + [self.bert_eos_id]
                turn_ids = [posts_turn_ids[0]] + history_turn_ids + resp_turns + [resp_turns[-1]]
                position_ids = [0] + history_position_ids + list(
                    range(history_position_ids[-1] + 1, history_position_ids[-1]+len(response)+2))    # 这里+2是因为最后还有eos_id
                src_ids = [0] + history_src_ids + [0] * (len(response) + 1)
                input_mask = [1] * len(input_ids)

                # 获取对应的单词标签
                # 在GPT内部计算loss时有shift操作，所以这里只需要逐一对齐即可
                lm_labels = [-1] * posts_len + [-1] + response[1:] + [self.bert_eos_id]

                assert len(input_ids) == len(position_ids)
                assert len(input_ids) == len(src_ids)

                padding = [0] * (self._max_sent_length - len(input_ids))
                input_ids = input_ids + padding
                token_type_ids = token_type_ids + padding
                turn_ids = turn_ids + padding
                position_ids = position_ids + padding
                src_ids = src_ids + padding
                input_mask = input_mask + padding
                lm_labels = lm_labels + [-1] * len(padding)

                all_input_ids.append(input_ids)
                all_token_type_ids.append(token_type_ids)
                all_turn_ids.append(turn_ids)
                all_position_ids.append(position_ids)
                all_source_ids.append(src_ids)
                all_input_mask.append(input_mask)
                all_lm_labels.append(lm_labels)
                posts_lens.append(posts_len)
            else:
                # 预测时输入是不包含response的
                # 对预测输入的封装不进行padding
                posts_len = len(posts_input_ids) + 1
                input_ids = [self.bert_go_id] + history_input_ids + [response[0]]
                token_type_ids = [self.bert_go_id] + history_token_type_ids + [resp_token_type_ids[0]]
                turn_ids = [posts_turn_ids[0]] + history_turn_ids + [resp_turns[0]]
                position_ids = [0] + history_position_ids + [history_position_ids[-1] + 1]
                src_ids = [0] + history_src_ids + [0]

                assert len(input_ids) == len(turn_ids)

                all_input_ids.append(input_ids)
                all_token_type_ids.append(token_type_ids)
                all_turn_ids.append(turn_ids)
                all_position_ids.append(position_ids)
                all_source_ids.append(src_ids)
                posts_lens.append(posts_len)


        # 将当前输入保存到词典中
        res["input_ids"] = all_input_ids
        res["token_type_ids"] = all_token_type_ids
        res["turn_ids"] = all_turn_ids
        res["position_ids"] = all_position_ids
        res["source_ids"] = all_source_ids
        res["input_mask"] = all_input_mask
        res["lm_labels"] = all_lm_labels
        res["posts_lens"] = posts_lens
        return res





    def merge_know_to_posts(self, know_pos, know_rt, posts, posts_turns, posts_token_type_ids):
        know_input_ids = []
        know_turn_ids = []
        know_token_type_ids = []
        know_position_ids = []
        posts_position_ids = []
        posts_src_ids = []

        turn_len = len(posts)  # 表示上下文轮次的数量
        next_turn_start_pid = 0
        for i in range(turn_len):
            st_tids = posts_turns[i]
            st_ttids = posts_token_type_ids[i]
            st_kpos = know_pos[i]
            st_krt = know_rt[i]
            st_len = len(st_tids)
            st_pids = list(range(next_turn_start_pid, next_turn_start_pid + st_len))
            next_turn_start_pid += st_len        # 下一轮position_id的起始位置
            posts_position_ids.append(st_pids)
            posts_src_ids.append([0] * st_len)
            # 对于当前位置中所有要插入知识的位置
            # 以及对应要插入的知识的relation和tail entity
            for pos, rt in zip(st_kpos, st_krt):
                # 获取当前要插入位置的position_id, turn_id, token_type_id
                position_id = st_pids[pos]
                turn_id = st_tids[pos]
                token_type_id = st_ttids[pos]

                relation_len = len(rt[0])
                tail_len = len(rt[1])
                know_input_ids.extend(rt[0] + rt[1])
                know_position_ids.extend([position_id + 1] * relation_len + [position_id + 2] * tail_len)

                know_turn_ids.extend([turn_id] * (relation_len + tail_len))
                know_token_type_ids.extend([token_type_id] * (relation_len + tail_len))

        # 根据max_know_length对知识进行截断
        know_input_ids = know_input_ids[-self._max_know_length:]
        know_turn_ids = know_turn_ids[-self._max_know_length:]
        know_token_type_ids = know_token_type_ids[-self._max_know_length:]
        know_position_ids = know_position_ids[-self._max_know_length:]
        know_src_ids = [1] * len(know_position_ids)

        return (know_input_ids, know_turn_ids, know_token_type_ids,
                know_position_ids, know_src_ids, posts_position_ids, posts_src_ids)


