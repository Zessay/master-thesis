from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import jieba
import torch
import numpy as np


class MyMetrics(object):
    def __init__(self):
        self.refs = []    # 表示正确结果语句，原始字符串，没有使用空格分隔
        self.hyps = []    # 表示预测结果语句，原始字符串，没有使用空格分隔

    def forword(self, ref, hyp):
        self.refs.append([jieba.lcut(ref)])
        self.hyps.append(jieba.lcut(hyp))

    def calc_distinct_k(self, k):
        d = {}
        tot = 0
        for sen in self.hyps:
            for i in range(0, len(sen)-k):
                key = ''.join(sen[i:i+k])
                d[key] = 1
                tot += 1
        return len(d) / tot

    def close(self):
        result = {}
        for i in range(1, 5):
            result["distict_%d" % i] = self.calc_distinct_k(i)
            try:
                weights = [1 / i] * i + [0] * (4 - i)
                result.update(
                    {"bleu-%d" % i: corpus_bleu(self.refs, self.hyps, weights, smoothing_function=SmoothingFunction().method3)})
            except ZeroDivisionError as _:
                result.update({"bleu-%d" % i: 0})

        return result


class MyPerplexity(object):
    def __init__(self, unk_id: int):
        self.unk_id = unk_id
        self.word_loss = 0
        self.length_sum = 0

    def forward(self, resp_length: int, resp_ids: torch.Tensor, gen_log_prob: torch.Tensor):
        """
        resp_ids: 一维的torch.Tensor, long类型，[resp_length, ]
        gen_log_prob: 二维的torch.Tensor, float类型, 经过log_softmax之后的，[decoder_len, vocab_size]
        注意这里的decoder_len是有可能小于resp_length的，所有要计算二者的较小值
        """
        # 计算较小的长度
        decoder_length = gen_log_prob.size(0)
        resp_len = min(resp_length, decoder_length)
        # 得到距离裁剪之后的结果
        resp_now = resp_ids[:resp_len]
        gen_now = gen_log_prob[:resp_len, :]

        # 计算常规的单词
        # [resp_len, ]
        normal_mask = (resp_now != self.unk_id).float()
        word_loss = - (gen_now.gather(-1, resp_now.unsqueeze(1))[:, 0] * resp_len).sum()
        length_sum = normal_mask.sum()
        # 这里不需要像原始的ppl那样，计算>=vocab_size对应的损失，因为这里不存在那样的情况

        self.word_loss += word_loss.tolist()
        self.length_sum += length_sum.tolist()

    def close(self):
        result = {}
        if self.length_sum <= 0:
            raise RuntimeError("The metric has not been forwarded correctly.")

        result.update({"perplexity": np.exp(self.word_loss / self.length_sum)})
        return result



