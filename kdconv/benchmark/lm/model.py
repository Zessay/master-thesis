# coding=utf-8
# @Author: 莫冉
# @Date: 2021-01-05
from __future__ import print_function
import os
import time
import json
import jieba
import logging
import numpy as np
import tensorflow as tf

from utils.output_projection import output_projection_layer, MyDense
from utils.summaryx_helper import SummaryHelper

logger = logging.getLogger(__name__)


class LM(object):
	def __init__(self, data, args, embed):
		# 定义输入的占位符
		self.posts = tf.placeholder(tf.int32, (None, None), 'enc_inps')  # batch*len
		self.posts_length = tf.placeholder(tf.int32, (None,), 'enc_lens')  # batch
		self.origin_responses = tf.placeholder(tf.int32, (None, None), 'dec_inps')  # batch*len
		self.origin_responses_length = tf.placeholder(tf.int32, (None,), 'dec_lens')  # batch
		self.is_train = tf.placeholder(tf.bool)

		# deal with original data to adapt encoder and decoder
		# 记录batch_size以及response的长度
		batch_size, decoder_len = tf.shape(self.origin_responses)[0], tf.shape(self.origin_responses)[1]
		# 定义输入和输出，对应的长度都是decoder_len-1
		# input是去除最后一个位置的token，target是去除第一个位置的token
		self.responses_input = tf.split(self.origin_responses, [decoder_len-1, 1], 1)[0]
		self.responses_target = tf.split(self.origin_responses, [1, decoder_len-1], 1)[1]
		self.responses_length = self.origin_responses_length - 1
		decoder_len = decoder_len - 1

		# 定义decoder的mask矩阵，[batch_size, decoder_len]
		self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length-1,
			decoder_len), reverse=True, axis=1), [-1, decoder_len])

		# initialize the training process
		# 初始化训练参数，学习率等
		self.learning_rate = tf.Variable(float(args.lr), trainable=False, dtype=tf.float32)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * args.lr_decay)
		self.global_step = tf.Variable(0, trainable=False)

		# build the embedding table and embedding input
		# 定义词向量矩阵
		if embed is None:
			# initialize the embedding randomly
			self.embed = tf.get_variable('embed', [data.vocab_size, args.embedding_size], tf.float32)
		else:
			# initialize the embedding by pre-trained word vectors
			self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)
		
		self.encoder_input = tf.nn.embedding_lookup(self.embed, self.posts)
		self.decoder_input = tf.nn.embedding_lookup(self.embed, self.responses_input)
		#self.decoder_input = tf.cond(self.is_train,
		#							 lambda: tf.nn.dropout(tf.nn.embedding_lookup(self.embed, self.responses_input), 0.8),
		#							 lambda: tf.nn.embedding_lookup(self.embed, self.responses_input))

		# build rnn_cell
		# 定义GRU单元，这里的eh_size表示隐层的单元数
		cell = tf.nn.rnn_cell.GRUCell(args.eh_size)

		# get output projection function
		# 映射到词表分布上
		output_fn = MyDense(data.vocab_size, use_bias=True)
		# dh_size表示解码器的隐层输出
		# vocab_size表示词表的大小
		sampled_sequence_loss = output_projection_layer(args.dh_size, data.vocab_size, args.softmax_samples)

		# build encoder
		# 这里主要用于测试阶段，通过对post编码得到隐向量
		# 如果是训练阶段，这里应该是全0的矩阵
		with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
			_, self.encoder_state = tf.nn.dynamic_rnn(cell, self.encoder_input,
				self.posts_length, dtype=tf.float32, scope="decoder_rnn")

		# construct helper and attention
		# 这里之所以填充eos_id，是因为在解码时，decoder的输入第一个字符就是<eos>
		# 这在对输入数据进行封装的时候就定义好了
		infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embed, tf.fill([batch_size], data.eos_id), data.eos_id)

		dec_start = tf.cond(self.is_train,
							lambda: tf.zeros([batch_size, args.dh_size], dtype=tf.float32),
							lambda: self.encoder_state)

		# build decoder (train)
		with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
			# 这里的Output是[batch_size, decoder_length, hidden_size]
			self.decoder_output, _ = tf.nn.dynamic_rnn(cell, self.decoder_input, self.responses_length,
												 dtype=tf.float32, initial_state=dec_start,
												 scope='decoder_rnn')
			#self.decoder_output = tf.nn.dropout(self.decoder_output, 0.8)
			# decoder_output的维度是[batch_size, decoder_length, hidden_size]
			# responses_target的维度是[batch_size, decoder_length]
			# decoder_mask的维度是[batch_size, decoder_mask]
			# 输出decoder_distribution_teacher的维度是[batch_size, decoder_len, vocab_size]
			# 输出的decoder_loss是0维的平均损失
			# 输出的decoder_all_loss是每一个样例一句话的损失[batch_size,]
			self.decoder_distribution_teacher, self.decoder_loss, self.decoder_all_loss = \
				sampled_sequence_loss(self.decoder_output, self.responses_target, self.decoder_mask)

		# build decoder (test)
		with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
			decoder_infer = tf.contrib.seq2seq.BasicDecoder(cell, infer_helper, dec_start, output_layer=output_fn)
			infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_infer, impute_finished=True,
																 maximum_iterations=args.max_decoder_length, scope="decoder_rnn")
			# [batch_size, max_decoder_length, vocab_size]
			self.decoder_distribution = infer_outputs.rnn_output
			# 得到生成的单词的索引[batch_size, max_decoder_length]
			# 这里由于前4个单词是["<pad>", "<unk>", "<go>", "<eos>"]
			# 所以要去除<pad>和<unk>这两个单词
			self.generation_index = tf.argmax(tf.split(self.decoder_distribution,
				[2, data.vocab_size-2], 2)[1], 2) + 2 # for removing UNK

		# calculate the gradient of parameters and update
		self.params = [k for k in tf.trainable_variables() if args.name in k.name]
		opt = tf.train.AdamOptimizer(self.learning_rate)
		gradients = tf.gradients(self.decoder_loss, self.params)
		clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
				args.grad_clip)
		self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
				global_step=self.global_step)

		# save checkpoint
		self.latest_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
				max_to_keep=args.checkpoint_max_to_keep, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
		self.best_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
				max_to_keep=1, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

		# create summary for tensorboard
		# 记录参数
		self.create_summary(args)

	def store_checkpoint(self, sess, path, key, name):
		if key == "latest":
			self.latest_saver.save(sess, path, global_step = self.global_step, latest_filename = name)
		else:
			self.best_saver.save(sess, path, global_step = self.global_step, latest_filename = name)
			#self.best_global_step = self.global_step

	def create_summary(self, args):
		self.summaryHelper = SummaryHelper("%s/%s_%s" % \
				(args.log_dir, args.name, time.strftime("%H%M%S", time.localtime())), args)

		self.trainSummary = self.summaryHelper.addGroup(scalar=["loss", "perplexity"], prefix="train")

		scalarlist = ["loss", "perplexity"]
		tensorlist = []
		textlist = []
		for i in args.show_sample:
			textlist.append("show_str%d" % i)
		self.devSummary = self.summaryHelper.addGroup(scalar=scalarlist, tensor=tensorlist, text=textlist,
													  prefix="dev")

	def print_parameters(self):
		for item in self.params:
			logger.info('%s: %s' % (item.name, item.get_shape()))
	
	def step_decoder(self, session, data, forward_only=False):
		input_feed = {self.posts: data['post'], self.posts_length: data['post_length'],
					  self.origin_responses: data['resp'], self.origin_responses_length: data['resp_length'],
					  self.is_train: True}
		if forward_only:
			output_feed = [self.decoder_loss, self.decoder_distribution_teacher, self.decoder_output]
		else:
			output_feed = [self.decoder_loss, self.gradient_norm, self.update]
		return session.run(output_feed, input_feed)

	def inference(self, session, data):
		input_feed = {self.posts: data['post'], self.posts_length: data['post_length'],
					  self.origin_responses: data['resp'], self.origin_responses_length: data['resp_length'],
					  self.is_train: False}
		output_feed = [self.generation_index, self.decoder_distribution_teacher, self.decoder_all_loss]
		return session.run(output_feed, input_feed)

	def evaluate(self, sess, data, batch_size, key_name):
		loss = np.zeros((1,))
		times = 0
		data.restart(key_name, batch_size=batch_size, shuffle=False)
		batched_data = data.get_next_batch(key_name)
		while batched_data != None:
			outputs = self.step_decoder(sess, batched_data, forward_only=True)
			loss += outputs[0]
			times += 1
			batched_data = data.get_next_batch(key_name)
		loss /= times

		logger.info(f'Evaluate loss: {loss:.2f} | perplexity on {key_name} set: {np.exp(loss): .2f}')
		return loss

	def train_process(self, sess, data, args):
		loss_step, time_step, epoch_step = np.zeros((1,)), .0, 0
		previous_losses = [1e18] * 3
		best_valid = 1e18
		data.restart("train", batch_size=args.batch_size, shuffle=True)
		batched_data = data.get_next_batch("train")

		# 显示第1个batch前两个样例的情况
		for i in range(2):
			logger.info(f"post@{i}: {data.convert_ids_to_tokens(batched_data['post_allvocabs'][i].tolist(), trim=False)}")
			logger.info(f"resp@{i}: {data.convert_ids_to_tokens(batched_data['resp_allvocabs'][i].tolist(), trim=False)}")

		for epoch_step in range(args.epochs):
			while batched_data != None:
				# 如果当前的step进行到需要打印的步骤，则进行打印
				if self.global_step.eval() % args.checkpoint_steps == 0 and self.global_step.eval() != 0:
					show = lambda a: '[%s]' % (' '.join(['%.2f' % x for x in a]))
					logger.info("Epoch %d global step %d learning rate %.4f step-time %.2f perplexity %s"
						  % (epoch_step, self.global_step.eval(), self.learning_rate.eval(),
							 time_step, show(np.exp(loss_step))))
					self.trainSummary(self.global_step.eval() // args.checkpoint_steps, {'loss': loss_step, 'perplexity': np.exp(loss_step)})
					self.store_checkpoint(sess, '%s/checkpoint_latest/%s' % (args.model_dir, args.name), "latest", args.name)

					dev_loss = self.evaluate(sess, data, args.batch_size, "dev")
					self.devSummary(self.global_step.eval() // args.checkpoint_steps, {'loss': dev_loss, 'perplexity': np.exp(dev_loss)})

					if np.sum(loss_step) > max(previous_losses):
						logger.info(f"当前的损失为 {loss_step} | 学习率衰减")
						sess.run(self.learning_rate_decay_op)
					# 保存最佳模型
					if dev_loss < best_valid:
						best_valid = dev_loss
						self.store_checkpoint(sess, '%s/checkpoint_best/%s' % (args.model_dir, args.name), "best", args.name)

					previous_losses = previous_losses[1:] + [np.sum(loss_step)]
					loss_step, time_step = np.zeros((1,)), .0

				# 计算损失
				start_time = time.time()
				loss_step += self.step_decoder(sess, batched_data)[0] / args.checkpoint_steps
				time_step += (time.time() - start_time) / args.checkpoint_steps
				batched_data = data.get_next_batch("train")

			data.restart("train", batch_size=args.batch_size, shuffle=True)
			batched_data = data.get_next_batch("train")

	def test_process_hits(self, sess, data, args):

		with open(os.path.join(args.datapath, 'test_distractors.json'), 'r', encoding="utf-8") as f:
			test_distractors = json.load(f)

		data.restart("test", batch_size=1, shuffle=False)
		batched_data = data.get_next_batch("test")

		loss_record = []  # 记录损失
		cnt = 0
		while batched_data != None:
			batched_data['resp_length'] = [len(batched_data['resp'][0])]
			batched_data['resp'] = batched_data['resp'].tolist()
			for each_resp in test_distractors[cnt]:
				# 得到负样本resp转化之后的索引，以及对应的长度
				batched_data['resp'].append([data.eos_id] + data.convert_tokens_to_ids(jieba.lcut(each_resp)) + [data.eos_id])
				batched_data['resp_length'].append(len(batched_data['resp'][-1]))
			max_length = max(batched_data['resp_length'])
			resp = np.zeros((len(batched_data['resp']), max_length), dtype=int)
			for i, each_resp in enumerate(batched_data['resp']):
				resp[i, :len(each_resp)] = each_resp
			batched_data['resp'] = resp

			post = []
			post_length = []
			for _ in range(len(resp)):
				post = post + batched_data['post'].tolist()
				post_length = post_length + batched_data['post_length'].tolist()
			batched_data['post'] = post               # [num_choices, max_post_length]
			batched_data['post_length'] = post_length # [num_choices, ]

			_, _, loss = self.inference(sess, batched_data)
			loss_record.append(loss)  # 这里的loss为[num_choices, ]
			cnt += 1

			batched_data = data.get_next_batch("test")

		assert cnt == len(test_distractors)

		loss = np.array(loss_record)
		# 这里表示对所有的正负样本的损失按照从小到大的顺序排序
		loss_rank = np.argsort(loss, axis=1)
		# 由于其中的第0个为正样本，所以计算第0个覆盖率
		hits1 = float(np.mean(loss_rank[:, 0] == 0))
		hits3 = float(np.mean(np.min(loss_rank[:, :3], axis=1) == 0))
		hits5 = float(np.mean(np.min(loss_rank[:, :5], axis=1) == 0))
		return {'hits@1' : hits1, 'hits@3': hits3, 'hits@5': hits5}

	def test_process(self, sess, data, args):
		# 这里首先获取预定义的指标
		metric1 = data.get_teacher_forcing_metric()   # 这里主要计算ppl指标
		metric2 = data.get_inference_metric()         # 这里主要计算bleu和distinct指标
		data.restart("test", batch_size=args.batch_size, shuffle=False)
		batched_data = data.get_next_batch("test")

		for i in range(2):
			logger.info(f"post@{i}: {data.convert_ids_to_tokens(batched_data['post_allvocabs'][i].tolist(), trim=False)}")
			logger.info(f"resp@{i}: {data.convert_ids_to_tokens(batched_data['resp_allvocabs'][i].tolist(), trim=False)}")

		results = []
		while batched_data != None:
			# 其中batched_responses_ids为[batch_size, max_decoder_length]
			# gen_log_prob为[batch_size, max_decoder_length, vocab_size]
			batched_responses_id, gen_log_prob, _ = self.inference(sess, batched_data)
			# 其中resp_allvocabs字段表示回复中所有单词的索引[batch_size, max_resp_length]
			# resp_length表示这个batch中所有回复的长度[batch_size, ]
			# gen_log_prob表示模型计算的对数概率 [batch_size, max_decoder_length, vocab_size]
			# 这3个键都是必须包含的，其中resp_allvocabs中max_resp_length和max_decoder_length可以不一样
			metric1_data = {'resp_allvocabs': np.array(batched_data['resp_allvocabs']),
							'resp_length': np.array(batched_data['resp_length']),
							'gen_log_prob': np.array(gen_log_prob)}
			metric1.forward(metric1_data)
			batch_results = []
			for response_id in batched_responses_id:
				result_token = []
				response_id_list = response_id.tolist()
				# 将单词的索引转化为id
				response_token = data.convert_ids_to_tokens(response_id_list)
				# 找到eos token对应的第一个位置，并截断
				if data.eos_id in response_id_list:
					result_id = response_id_list[:response_id_list.index(data.eos_id)+1]
				else:
					result_id = response_id_list
				# 找到第一个eos，并截断
				for token in response_token:
					if token != data.ext_vocab[data.eos_id]:
						result_token.append(token)
					else:
						break
				results.append(result_token)
				batch_results.append(result_id)

			metric2_data = {'gen': np.array(batch_results),
							'post_allvocabs': np.array(batched_data['post_allvocabs']),
							'resp_allvocabs':np.array(batched_data['resp_allvocabs']),}
			metric2.forward(metric2_data)
			batched_data = data.get_next_batch("test")

		# 计算ppl, bleu以及distinct指标
		res = metric1.close()
		res.update(metric2.close())
		# 将正负样本都经过模型，观察正样本的损失是不是最小
		res.update(self.test_process_hits(sess, data, args))

		test_file = args.output_dir + "/%s_%s.txt" % (args.name, "test")
		with open(test_file, 'w', encoding="utf-8") as f:
			print("Test Result:")
			res_print = list(res.items())
			res_print.sort(key=lambda x: x[0])
			for key, value in res_print:
				if isinstance(value, float):
					print("\t%s:\t%f" % (key, value))
					f.write("%s:\t%f\n" % (key, value))
			f.write('\n')
			for i in range(len(res['post'])):
				f.write("resp:\t%s\n" % " ".join(res['resp'][i]))
				f.write("gen:\t%s\n\n" % " ".join(res['gen'][i]))


		logger.info("result output to %s." % test_file)
		return {key: val for key, val in res.items() if type(val) in [bytes, int, float]}

