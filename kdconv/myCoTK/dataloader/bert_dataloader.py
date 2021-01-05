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

from cotk.metric import MetricChain, SingleTurnDialogRecorder
from ..metric import BleuCorpusMetric, SingleTurnDistinct
from transformers import BertTokenizer
# from pytorch_pretrained_bert.tokenization import BertTokenizer

import jieba
from gensim.summarization import bm25


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
                    post_sent = (post_sent + [nxt_sent])[- self.num_turns + 1:]
                    corpus_post_sent = corpus_post_sent + [sentence]

                    i += 1

            logger.info(f"{key}文件共获取有效数据{len(corpus_post)}条")

            logger.info("采样负样本用于训练模型")
            # 当前文件的干扰选项
            distractor_file = os.path.join(self._file_path, '%s_distractors.json' % key)
            # 如果存在当前文件的干扰选项（即负样本）
            if os.path.exists(distractor_file):
                with open(distractor_file) as f:
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

                with open(distractor_file, 'w') as f:
                    json.dump(origin_data[key]['resp_distractors'], f, ensure_ascii=False, indent=4)

        logger.info(f"完成数据读取和负采样，共计用时%f ms，准备构建词表" % (time.time() - begin_time))
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
                while i + 1 < len(messages):
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
                with open(distractor_file) as f:
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

                with open(distractor_file, 'w') as f:
                    json.dump(origin_data[key]['resp_distractors'], f, ensure_ascii=False, indent=4)

        logger.info(f"完成数据读取和负采样，共计用时%f ms，准备构建词表" % (time.time() - begin_time))
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
            # 这里减去3因为一个[CLS]和两个[SEP]
            post_max_length = self._max_sent_length - 3 - resp_max_length
            # 选择历史回复中后面的靠后的单词
            post_bert = list(chain(*post_bert))[-post_max_length:]

            for tt, (option_id, option) in enumerate(zip(option_ids, options)):
                can_resps.append(ori_can_resps[option_id])

                resp_id = [self.bert_go_id] + option[: single_resp_max_length] + [self.bert_eos_id]
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
