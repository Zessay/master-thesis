import os
import time
from collections import Counter
from itertools import chain
import multiprocessing
from multiprocessing import Pool
import tqdm

import jieba
import json

import numpy as np

from cotk._utils.file_utils import get_resource_file_path
from cotk.dataloader import LanguageProcessingBase, BERTLanguageProcessingBase, SingleTurnDialog
from cotk.metric import MetricChain, PerplexityMetric, SingleTurnDialogRecorder
from ..metric import SingleTurnResponseRecorder, BleuCorpusMetric, SingleTurnDistinct


class MyLM(SingleTurnDialog):
    """
    语言模型
    train和dev的输入形式为："post"字段为空，"resp"表示每一轮对话句子之间的连接，每一句话之间用"<eos>"连接
    test的输入形式为："post"字段为上下文语句，"resp"表示当前语句
    min_vocab_times: int型，表示单词出现的最少次数
    invalid_vocab_times: int型
    """
    def __init__(self, file_id="../data/film", min_vocab_times=0,
                 max_sent_length=10086, invalid_vocab_times=0):
        self._file_id = file_id
        self._file_path = get_resource_file_path(file_id)
        self._min_vocab_times = min_vocab_times
        self._max_sent_length = max_sent_length
        self._invalid_vocab_times = invalid_vocab_times
        super(MyLM, self).__init__()


    def _load_data(self):
        r'''Loading dataset, invoked during the initialization of :class:`SingleTurnDialog`.
        '''
        origin_data = {}
        vocab = {}

        def count_token(tokens):
            for token in tokens:
                vocab[token] = vocab[token] + 1 if token in vocab else 1

        for key in self.key_name:
            # post表示历史对话
            # resp表示回复
            origin_data[key] = {'post': [], 'resp': []}
            datas = json.load(open("%s/%s.json" % (self._file_path, key), "r", encoding="utf-8"))

            for data in datas:
                messages = data['messages']
                i = 0
                # 如果是训练集或者验证集
                if key != 'test':
                    resp_sent = []
                    while i + 1 < len(messages):

                        if i == 0:
                            # 对当前语句分词
                            tmp_sent = jieba.lcut(messages[i]['message'])
                            resp_sent = tmp_sent

                            count_token(tmp_sent)
                            if 'attrs' in messages[0]:
                                for attr in messages[0]['attrs']:
                                    h = jieba.lcut(attr['name'])
                                    r = jieba.lcut(attr['attrname'])
                                    t = jieba.lcut(attr['attrvalue'])
                                    # 统计知识库的单词
                                    count_token(h + r + t)

                        nxt_sent = jieba.lcut(messages[i + 1]['message'])
                        resp_sent = resp_sent + ['<eos>'] + nxt_sent

                        count_token(nxt_sent)
                        if 'attrs' in messages[i + 1]:
                            for attr in messages[i + 1]['attrs']:
                                h = jieba.lcut(attr['name'])
                                r = jieba.lcut(attr['attrname'])
                                t = jieba.lcut(attr['attrvalue'])
                                # 统计知识库的单词
                                count_token(h + r + t)

                        i += 1

                    # post一致都是空的
                    origin_data[key]['post'].append([])
                    # resp则是历史对话和当前回复的拼接
                    origin_data[key]['resp'].append(resp_sent)

                else:
                    post_sent = []
                    while i + 1 < len(messages):
                        # 如果是第一句话，则单独处理作为历史对话
                        if i == 0:
                            post_sent = jieba.lcut(messages[0]['message'])
                            count_token(post_sent)
                            if 'attrs' in messages[0]:
                                for attr in messages[0]['attrs']:
                                    h = jieba.lcut(attr['name'])
                                    r = jieba.lcut(attr['attrname'])
                                    t = jieba.lcut(attr['attrvalue'])
                                    # 统计知识库的单词
                                    count_token(h + r + t)

                        # 将下一句话作为回复
                        nxt_sent = jieba.lcut(messages[i + 1]['message'])
                        origin_data[key]['post'].append(post_sent)
                        origin_data[key]['resp'].append(nxt_sent)

                        # 将历史输入和当前的回复拼接作为下一段的历史对话
                        post_sent = post_sent + ['<eos>'] + nxt_sent

                        count_token(nxt_sent)
                        if 'attrs' in messages[i + 1]:
                            for attr in messages[i + 1]['attrs']:
                                h = jieba.lcut(attr['name'])
                                r = jieba.lcut(attr['attrname'])
                                t = jieba.lcut(attr['attrvalue'])
                                # 统计知识库单词
                                count_token(h + r + t)

                        i += 1

        # Important: Sort the words preventing the index changes between different runs
        # 按照单词出现的频率从高到低排序
        vocab = sorted(list(vocab.items()), key=lambda pair: (-pair[1], pair[0]))
        # 过滤掉出现频次过少的单词
        left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
        left_vocab = list(map(lambda x: x[0], left_vocab))
        # 形成最终的词表
        vocab_list = self.ext_vocab + left_vocab
        valid_vocab_len = len(vocab_list)        # int型，表示词表中单词的数量
        valid_vocab_set = set(vocab_list)        # 对词表进行去重

        # 找出不存在词表中，但是大于有效次数的单词
        left_vocab = list(filter(lambda x: x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, vocab))
        # 最终的词表
        vocab_list.extend(list(map(lambda x: x[0], left_vocab)))

        print("valid vocab list length = %d" % valid_vocab_len)
        print("vocab list length = %d" % len(vocab_list))

        # 单词到id的映射
        word2id = {w: i for i, w in enumerate(vocab_list)}
        line2id = lambda line: ([self.go_id] +
                                  list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) +
                                  [self.eos_id])[:self._max_sent_length]
        line2id_post = lambda line: ([self.go_id] +
                                  list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line))
                                  )[:self._max_sent_length]
        line2id_resp = lambda line: ([self.eos_id] +
                                  list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) +
                                  [self.eos_id])[:self._max_sent_length]

        data = {}
        data_size = {}
        # 将单词转化为id
        for key in self.key_name:
            data[key] = {}
            # 如果是训练集和验证集
            if key != 'test':
                data[key]['post'] = list(map(line2id, origin_data[key]['post']))
                data[key]['resp'] = list(map(line2id, origin_data[key]['resp']))
            # 如果是测试集
            else:
                data[key]['post'] = list(map(line2id_post, origin_data[key]['post']))
                data[key]['resp'] = list(map(line2id_resp, origin_data[key]['resp']))
            # 上下文的长度
            data_size[key] = len(data[key]['post'])
            # 当前语句中的单词
            vocab = list(chain(*(origin_data[key]['post'] + origin_data[key]['resp'])))
            # 当前语句中单词的数量
            vocab_num = len(vocab)
            # 单词中oov单词的数量
            oov_num = len(list(filter(lambda word: word not in word2id, vocab)))
            # 不在有效单词集合中的单词的数量，减去oov单词的数量
            invalid_num = len(list(filter(lambda word: word not in valid_vocab_set, vocab))) - oov_num
            # 长度
            length = list(map(len, origin_data[key]['post'] + origin_data[key]['resp']))
            # 当前语句的长度 - 句子最大长度 + 1， 表示截断单词的总长度
            cut_num = np.sum(np.maximum(np.array(length) - self._max_sent_length + 1, 0))
            print("%s set. invalid rate: %f, unknown rate: %f, max length before cut: %d, cut word rate: %f" % \
                  (key, invalid_num / vocab_num, oov_num / vocab_num, max(length), cut_num / vocab_num))

        return vocab_list, valid_vocab_len, data, data_size


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
        metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key, reference_allvocabs_key="resp_allvocabs"))
        metric.add_metric(SingleTurnDialogRecorder(self, gen_key=gen_key))
        metric.add_metric(SingleTurnDistinct(self, gen_key=gen_key))
        return metric


class MySeq2Seq(SingleTurnDialog):
    """
    Seq2Seq模型
    输入为： "post"字段为 turn-1 <eos> <go> turn_2 <eos> <go> ....<eos> <go> last_turn
            "resp"字段为 回复的语句
    """
    def __init__(self, file_id="../data/film", min_vocab_times=0, num_turns=8,
            max_sent_length=10086, invalid_vocab_times=0):
        self._file_id = file_id
        self._file_path = get_resource_file_path(file_id)
        self._min_vocab_times = min_vocab_times
        self._max_sent_length = max_sent_length
        self._invalid_vocab_times = invalid_vocab_times
        self._num_turns = num_turns
        super(MySeq2Seq, self).__init__()

    def _load_data(self):
        r'''Loading dataset, invoked during the initialization of :class:`SingleTurnDialog`.
        '''
        origin_data = {}
        vocab = {}

        def count_token(tokens):
            for token in tokens:
                vocab[token] = vocab[token] + 1 if token in vocab else 1

        for key in self.key_name:
            origin_data[key] = {'post': [], 'resp': []}
            datas = json.load(open("%s/%s.json" % (self._file_path, key), "r", encoding="utf-8"))

            for data in datas:
                history = []
                messages = data['messages']
                i = 0
                nxt_sent = jieba.lcut(messages[0]['message'])
                while i + 1 < len(messages):
                    # 对句子进行分词
                    tmp_sent, nxt_sent = nxt_sent, jieba.lcut(messages[i + 1]['message'])
                    # 添加到历史对话中
                    history.append(tmp_sent)
                    post = []
                    # 从最大允许的轮次第一轮开始，往后获取前num_turns - 1轮作为历史轮
                    for jj in range(max(-self._num_turns+1, -i-1), 0):
                        post = post + history[jj] + ['<eos>', '<go>']
                    # 这里去除历史轮最后的两个标记符 '<eos>' 和 '<go>'
                    post = post[:-2]
                    origin_data[key]['post'].append(post)
                    origin_data[key]['resp'].append(nxt_sent)
                    # 计算当前语句中的单词数
                    count_token(nxt_sent)
                    if 'attrs' in messages[i + 1]:
                        # 计算知识库中的单词
                        for attr in messages[i + 1]['attrs']:
                            h = jieba.lcut(attr['name'])
                            r = jieba.lcut(attr['attrname'])
                            t = jieba.lcut(attr['attrvalue'])
                            count_token(h + r + t)

                    # 如果是第一句话，则需要统计第一句话以及其包含的知识库中的单词
                    if i == 0:
                        count_token(tmp_sent)
                        if 'attrs' in messages[0]:
                            for attr in messages[0]['attrs']:
                                h = jieba.lcut(attr['name'])
                                r = jieba.lcut(attr['attrname'])
                                t = jieba.lcut(attr['attrvalue'])
                                count_token(h + r + t)

                    i += 1

        # Important: Sort the words preventing the index changes between different runs
        # 按照单词出现的次数从高到低排序
        vocab = sorted(list(vocab.items()), key=lambda pair: (-pair[1], pair[0]))
        # 删除出现次数太少的单词
        left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
        left_vocab = list(map(lambda x: x[0], left_vocab))
        # 将额外添加的单词和过滤之后得到的单词合并
        vocab_list = self.ext_vocab + left_vocab
        valid_vocab_len = len(vocab_list)
        valid_vocab_set = set(vocab_list)

        # 在原始的词表中查找超过无效次数的单词并且没有在有效词表中
        left_vocab = list(filter(lambda x: x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, vocab))
        vocab_list.extend(list(map(lambda x: x[0], left_vocab)))

        print("valid vocab list length = %d" % valid_vocab_len)
        print("vocab list length = %d" % len(vocab_list))

        # 单词到索引的映射
        word2id = {w: i for i, w in enumerate(vocab_list)}
        line2id = lambda line: ([self.go_id] +
                    list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) +
                    [self.eos_id])[:self._max_sent_length]

        data = {}
        data_size = {}
        for key in self.key_name:
            data[key] = {}

            # 将单词转化为索引
            data[key]['post'] = list(map(line2id, origin_data[key]['post']))
            data[key]['resp'] = list(map(line2id, origin_data[key]['resp']))
            data_size[key] = len(data[key]['post'])

            # 根据数据中的分词结果汇总成词表
            vocab = list(chain(*(origin_data[key]['post'] + origin_data[key]['resp'])))
            vocab_num = len(vocab)     # 计算单词词表的大小
            # 计算oov单词的数量
            oov_num = len(list(filter(lambda word: word not in word2id, vocab)))
            # 除了oov单词和有效单词之外，其他单词
            invalid_num = len(list(filter(lambda word: word not in valid_vocab_set, vocab))) - oov_num
            # 计算post和resp中所有语句的长度
            length = list(map(len, origin_data[key]['post'] + origin_data[key]['resp']))
            # 计算需要截断单词的和
            cut_num = np.sum(np.maximum(np.array(length) - self._max_sent_length + 1, 0))
            print("%s set. invalid rate: %f, unknown rate: %f, max length before cut: %d, cut word rate: %f" %
                  (key, invalid_num / vocab_num, oov_num / vocab_num, max(length), cut_num / vocab_num))

        return vocab_list, valid_vocab_len, data, data_size

    def get_batch(self, key, indexes):
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        res = {}
        batch_size = len(indexes)
        # 计算整个上下文句子的长度
        res["post_length"] = np.array(list(map(lambda i: len(self.data[key]['post'][i]), indexes)))
        # 计算找到post中除了最后一轮之后前面的最后一个eos的索引，得到除去当前轮之外前面所有轮次的长度
        res["prev_length"] = np.array(list(map(lambda i: (
            len(self.data[key]['post'][i]) - self.data[key]['post'][i][::-1].index(self.eos_id, 1)
            if self.eos_id in self.data[key]['post'][i][:-1] else 0), indexes)))
        # 计算回复的长度
        res["resp_length"] = np.array(list(map(lambda i: len(self.data[key]['resp'][i]), indexes)))
        # post表示历史对话，维度为 [batch_size, post_length]
        # resp表示当前回复，维度为 [batch_size, resp_length]
        res_post = res["post"] = np.zeros((batch_size, np.max(res["post_length"])), dtype=int)
        res_resp = res["resp"] = np.zeros((batch_size, np.max(res["resp_length"])), dtype=int)
        for i, idx in enumerate(indexes):
            post = self.data[key]['post'][idx]
            resp = self.data[key]['resp'][idx]
            res_post[i, :len(post)] = post
            res_resp[i, :len(resp)] = resp

        res["post_allvocabs"] = res_post.copy()
        res["resp_allvocabs"] = res_resp.copy()
        res_post[res_post >= self.valid_vocab_len] = self.unk_id
        res_resp[res_resp >= self.valid_vocab_len] = self.unk_id

        return res

    def get_inference_metric(self, gen_key="gen"):
        '''Get metric for inference.

        It contains:

        * :class:`.metric.BleuCorpusMetric`
        * :class:`.metric.MultiTurnDialogRecorder`

        Arguments:
            gen_key (str): The key of generated sentences in index form.
                Refer to :class:`.metric.BleuCorpusMetric` or :class:`.metric.MultiTurnDialogRecorder`.
                Default: ``gen``.

        Returns:
            A :class:`.metric.MetricChain` object.
        '''
        metric = MetricChain()
        metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key, reference_allvocabs_key="resp_allvocabs"))
        metric.add_metric(SingleTurnResponseRecorder(self, gen_key=gen_key))
        metric.add_metric(SingleTurnDistinct(self, gen_key=gen_key))
        return metric


class MyMemSeq2Seq(SingleTurnDialog):
    """
    包含知识的Seq2Seq
    输入的数据：
    "post"字段为 turn-1 <eos> <go> turn_2 <eos> <go> ....<eos> <go> last_turn
    "resp"字段为 回复的语句
    "kg"表示当前这段对话中包含的所有知识，类型为List[Tuple[Tuple[str]]]
    "kg_index"表示当前这轮对话中使用的知识的索引List[int]
    """
    def __init__(self, file_id="../data/film", min_vocab_times=0, max_sent_length=10086, invalid_vocab_times=0, num_turns=8,
                 max_know_length=100):
        self._file_id = file_id
        self._file_path = get_resource_file_path(file_id)
        self._min_vocab_times = min_vocab_times
        self._max_sent_length = max_sent_length
        self._invalid_vocab_times = invalid_vocab_times
        self._num_turns = num_turns
        self._max_know_length = max_know_length
        super(MyMemSeq2Seq, self).__init__()

    def _load_data(self):
        r'''Loading dataset, invoked during the initialization of :class:`SingleTurnDialog`.
        '''
        origin_data = {}
        vocab = {}

        def count_token(tokens):
            for token in tokens:
                vocab[token] = vocab[token] + 1 if token in vocab else 1

        for key in self.key_name:
            origin_data[key] = {'post': [], 'resp': [], 'kg': [], 'kg_index': []}
            datas = json.load(open("%s/%s.json" % (self._file_path, key), "r", encoding='utf-8'))

            for data in datas:
                messages = data['messages']
                kg = []         # 保存知识的元组，具体为List[Tuple[Tuple[str]]]，h/r/t都是一个元组，元组中每个元素是分词之后的单词
                kg_index = []   # 保存每一句话对应知识的索引，List[List[int]]，有时候一句话对应多个知识
                kg_dict = {}    # 保存知识到索引的映射
                # 保存当前对话中的所有知识
                for message in messages:
                    kg_index.append([])
                    if 'attrs' in message:
                        for attr in message['attrs']:
                            h = jieba.lcut(attr['name'])
                            r = jieba.lcut(attr['attrname'])
                            t = jieba.lcut(attr['attrvalue'])
                            k = tuple((tuple(h), tuple(r), tuple(t)))
                            if k not in kg_dict:
                                kg_dict[k] = len(kg)
                                kg.append(k)
                            kg_index[-1].append(kg_dict[k])
                            # 统计知识中单词的数量
                            count_token(h + r + t)

                history = []
                i = 0
                nxt_sent = jieba.lcut(messages[0]['message'])
                while i + 1 < len(messages):
                    tmp_sent, nxt_sent = nxt_sent, jieba.lcut(messages[i + 1]['message'])
                    history.append(tmp_sent)
                    post = []
                    for jj in range(max(-self._num_turns + 1, -i - 1), 0):
                        post = post + history[jj] + ['<eos>', '<go>']
                    post = post[:-2]

                    count_token(nxt_sent)
                    if i == 0:
                        count_token(tmp_sent)

                    origin_data[key]['post'].append(post)
                    origin_data[key]['resp'].append(nxt_sent)
                    origin_data[key]['kg'].append(kg)
                    origin_data[key]['kg_index'].append(kg_index[i + 1])

                    i += 1

        # Important: Sort the words preventing the index changes between different runs
        # 构建词表
        vocab = sorted(list(vocab.items()), key=lambda pair: (-pair[1], pair[0]))
        left_vocab = list(filter(lambda x: x[1] >= self._min_vocab_times, vocab))
        left_vocab = list(map(lambda x: x[0], left_vocab))
        vocab_list = self.ext_vocab + left_vocab
        valid_vocab_len = len(vocab_list)
        valid_vocab_set = set(vocab_list)

        left_vocab = list(filter(lambda x: x[1] >= self._invalid_vocab_times and x[0] not in valid_vocab_set, vocab))
        vocab_list.extend(list(map(lambda x: x[0], left_vocab)))

        print("valid vocab list length = %d" % valid_vocab_len)
        print("vocab list length = %d" % len(vocab_list))

        # 单词到id的索引
        word2id = {w: i for i, w in enumerate(vocab_list)}
        know2id = lambda line: list(map(lambda word: word2id.get(word, self.unk_id), line))
        line2id = lambda line: ([self.go_id] + list(map(lambda word: word2id[word] if word in word2id else self.unk_id, line)) + [
            self.eos_id])[:self._max_sent_length]
        knows2id = lambda lines: list(map(know2id, lines))

        data = {}
        data_size = {}
        for key in self.key_name:
            data[key] = {}
            data[key]['post'] = list(map(line2id, origin_data[key]['post']))
            data[key]['resp'] = list(map(line2id, origin_data[key]['resp']))
            data[key]['kg'] = [list(map(knows2id, kg)) for kg in origin_data[key]['kg']]
            data[key]['kg_index'] = origin_data[key]['kg_index']
            data_size[key] = len(data[key]['post'])

        return vocab_list, valid_vocab_len, data, data_size


    def get_batch(self, key, indexes):
        if key not in self.key_name:
            raise ValueError("No set named %s." % key)
        res = {}
        batch_size = len(indexes)

        # 这一段和上面一样，计算当前batch中的最大长度，同时使用numpy封装数据
        res["post_length"] = np.array(list(map(lambda i: len(self.data[key]['post'][i]), indexes)))
        res["prev_length"] = np.array(list(map(lambda i: (
            len(self.data[key]['post'][i]) - self.data[key]['post'][i][::-1].index(self.eos_id, 1)
            if self.eos_id in self.data[key]['post'][i][:-1] else 0), indexes)))
        res["resp_length"] = np.array(list(map(lambda i: len(self.data[key]['resp'][i]), indexes)))
        res_post = res["post"] = np.zeros((batch_size, np.max(res["post_length"])), dtype=int)
        res_resp = res["resp"] = np.zeros((batch_size, np.max(res["resp_length"])), dtype=int)
        for i, idx in enumerate(indexes):
            post = self.data[key]['post'][idx]
            resp = self.data[key]['resp'][idx]
            res_post[i, :len(post)] = post
            res_resp[i, :len(resp)] = resp

        res["post_allvocabs"] = res_post.copy()
        res["resp_allvocabs"] = res_resp.copy()
        res_post[res_post >= self.valid_vocab_len] = self.unk_id
        res_resp[res_resp >= self.valid_vocab_len] = self.unk_id

        # 这一段封装知识
        # 计算所有对话中，知识最多的对话的知识数量
        max_kg_num = max([len(self.data[key]['kg'][idx]) for idx in indexes])
        res["kg_h_length"] = np.zeros((batch_size, max_kg_num), dtype=int)
        res["kg_hr_length"] = np.zeros((batch_size, max_kg_num), dtype=int)
        res['kg_hrt_length'] = np.zeros((batch_size, max_kg_num), dtype=int)
        for i, idx in enumerate(indexes):
            # 分别计算每一段对话中h, hr, hrt的长度
            kg_h_length = [min(self._max_know_length, len(sent[0])) for sent in self.data[key]['kg'][idx]]
            res["kg_h_length"][i, :len(kg_h_length)] = kg_h_length
            kg_hr_length = [min(self._max_know_length, len(sent[0]) + len(sent[1])) for sent in self.data[key]['kg'][idx]]
            res["kg_hr_length"][i, :len(kg_hr_length)] = kg_hr_length
            kg_hrt_length = [min(self._max_know_length, len(sent[0]) + len(sent[1]) + len(sent[2])) for sent in
                             self.data[key]['kg'][idx]]
            res["kg_hrt_length"][i, :len(kg_hrt_length)] = kg_hrt_length

        # 知识的维度 batch_size * max_kg_num * kg_hrt_length
        res['kg'] = np.zeros((batch_size, max_kg_num, np.max(res["kg_hrt_length"])), dtype=int)
        for i, idx in enumerate(indexes):
            for j, tri in enumerate(self.data[key]['kg'][idx]):
                sent = (tri[0] + tri[1] + tri[2])[:self._max_know_length]
                res['kg'][i, j, :len(sent)] = sent

        # batch_size * max_kg_num
        # 将每一句话对应的知识点位置置为1，其他位置为0
        res['kg_index'] = np.zeros((batch_size, max_kg_num), dtype=float)
        for i, idx in enumerate(indexes):
            for kgid in self.data[key]['kg_index'][idx]:
                res['kg_index'][i, kgid] = 1

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
        metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key, reference_allvocabs_key="resp_allvocabs"))
        metric.add_metric(SingleTurnResponseRecorder(self, gen_key=gen_key))
        metric.add_metric(SingleTurnDistinct(self, gen_key=gen_key))
        return metric