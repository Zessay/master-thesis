# coding=utf-8
# @Author: 莫冉
# @Date: 2021-01-07
import numpy as np
import tensorflow as tf
import time

from utils.output_projection import output_projection_layer, MyDense, MyAttention
from utils import SummaryHelper
import jieba
import os
import json
import logging

logger = logging.getLogger(__name__)


class Seq2SeqModel(object):
	def __init__(self, data, args, embed):
		# posts表示编码器，即历史对话输入 [batch, encoder_len]
		# posts_length表示输入的每一句话的实际长度 [batch]
		# prev_length除去最后一轮，之前轮次语句的长度（包含<go>和<eos>），[batch]
		self.posts = tf.placeholder(tf.int32, (None, None), 'enc_inps')  # batch*len
		self.posts_length = tf.placeholder(tf.int32, (None,), 'enc_lens')  # batch
		self.prevs_length = tf.placeholder(tf.int32, (None,), 'enc_lens_prev')  # batch

		# origin_responses表示回复的内容，[batch, resp_len]
		# origin_responses_length表示每一个回复的实际长度，[batch, ]
		self.origin_responses = tf.placeholder(tf.int32, (None, None), 'dec_inps')  # batch*len
		self.origin_responses_length = tf.placeholder(tf.int32, (None,), 'dec_lens')  # batch
		self.is_train = tf.placeholder(tf.bool)

		# deal with original data to adapt encoder and decoder
		batch_size, decoder_len = tf.shape(self.origin_responses)[0], tf.shape(self.origin_responses)[1]
		# 这里对回复进行分割，此时祛除了回复中的go_id
		self.responses = tf.split(self.origin_responses, [1, decoder_len-1], 1)[1] # no go_id
		self.responses_length = self.origin_responses_length - 1
		# 这里得到解码器的输入和输出，输入去除了最后的eos_id，输出去除了最开始的go_id，这样保证对齐
		# [batch, decoder_len]（这里的decoder_len等于resp_len-1）
		self.responses_input = tf.split(self.origin_responses, [decoder_len-1, 1], 1)[0] # no eos_id
		self.responses_target = self.responses
		decoder_len = decoder_len - 1
		# 编码器输入 [batch, encoder_len]
		self.posts_input = self.posts   # batch*len
		# 这里计算decoder的mask矩阵
		# 等于[batch, decoder_len]
		self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length-1,
			decoder_len), reverse=True, axis=1), [-1, decoder_len])

		# initialize the training process
		self.learning_rate = tf.Variable(float(args.lr), trainable=False, dtype=tf.float32)
		self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * args.lr_decay)
		self.global_step = tf.Variable(0, trainable=False)

		# build the embedding table and embedding input
		if embed is None:
			# initialize the embedding randomly
			self.embed = tf.get_variable('embed', [data.vocab_size, args.embedding_size], tf.float32)
		else:
			# initialize the embedding by pre-trained word vectors
			self.embed = tf.get_variable('embed', dtype=tf.float32, initializer=embed)

		# 将编码器和解码器的输入转化为词向量
		# encoder_input: [batch, encoder_len, embed_size]
		# decoder_input: [batch, decoder_len, embed_size]
		self.encoder_input = tf.nn.embedding_lookup(self.embed, self.posts)
		self.decoder_input = tf.nn.embedding_lookup(self.embed, self.responses_input)
		#self.encoder_input = tf.cond(self.is_train,
		#							 lambda: tf.nn.dropout(tf.nn.embedding_lookup(self.embed, self.posts_input), 0.8),
		#							 lambda: tf.nn.embedding_lookup(self.embed, self.posts_input)) #batch*len*unit
		#self.decoder_input = tf.cond(self.is_train,
		#							 lambda: tf.nn.dropout(tf.nn.embedding_lookup(self.embed, self.responses_input), 0.8),
		#							 lambda: tf.nn.embedding_lookup(self.embed, self.responses_input))

		# build rnn_cell
		cell_enc = tf.nn.rnn_cell.GRUCell(args.eh_size)
		cell_dec = tf.nn.rnn_cell.GRUCell(args.dh_size)

		# build encoder
		with tf.variable_scope('encoder'):
			# encoder_output: [batch, encoder_len, eh_size]
			# encoder_state: [batch, eh_size]
			encoder_output, encoder_state = tf.nn.dynamic_rnn(cell_enc, self.encoder_input,
				self.posts_length, dtype=tf.float32, scope="encoder_rnn")

		# get output projection function
		output_fn = MyDense(data.vocab_size, use_bias = True)
		sampled_sequence_loss = output_projection_layer(args.dh_size, data.vocab_size, args.softmax_samples)

		encoder_len = tf.shape(encoder_output)[1]
		# 这里计算posts和prevs的mask矩阵
		posts_mask = tf.sequence_mask(self.posts_length, encoder_len)
		prevs_mask = tf.sequence_mask(self.prevs_length, encoder_len)
		# 不同为1，相同为1
		# 这里表示只关注最后一轮，[batch, encoder_len]
		attention_mask = tf.reshape(tf.logical_xor(posts_mask, prevs_mask), [batch_size, encoder_len])

		# construct helper and attention
		train_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_input, self.responses_length)
		# 这里在推理的时候，起始位置全部使用go_id进行填充
		# 这在对输入数据进行封装时即进行了定义
		infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embed, tf.fill([batch_size], data.go_id), data.eos_id)

		# 这里编码器是按照多轮输入进行编码的
		# 但是解码器在attention的时候只关注最后一轮输入
		# 这里定义输入输出attention
		attn_mechanism = MyAttention(args.dh_size, encoder_output, attention_mask)
		cell_dec_attn = tf.contrib.seq2seq.AttentionWrapper(cell_dec, attn_mechanism,
															attention_layer_size=args.dh_size)
		# 把编码器最后一层的隐状态映射到解码器隐状态的维度
		# [batch, dh_size]
		enc_state_shaping = tf.layers.dense(encoder_state, args.dh_size, activation = None)
		dec_start = cell_dec_attn.zero_state(batch_size, dtype = tf.float32).clone(cell_state = enc_state_shaping)

		# build decoder (train)
		with tf.variable_scope('decoder'):
			decoder_train = tf.contrib.seq2seq.BasicDecoder(cell_dec_attn, train_helper, dec_start)
			train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_train, impute_finished = True, scope = "decoder_rnn")
			self.decoder_output = train_outputs.rnn_output
			#self.decoder_output = tf.nn.dropout(self.decoder_output, 0.8)
			# 计算损失和概率分布
			# decoder_distribution_teacher：[batch, decoder_length, vocab_size] （这里都是对数概率）
			# decoder_loss，基于这个batch中所有词的损失，0维
			# decoder_all_loss，每一句话的损失，[batch, ]
			self.decoder_distribution_teacher, self.decoder_loss, self.decoder_all_loss = \
				sampled_sequence_loss(self.decoder_output, self.responses_target, self.decoder_mask)

		# build decoder (test)
		with tf.variable_scope('decoder', reuse=True):
			# 这里output_fn会重用上面的权重和偏置
			decoder_infer = tf.contrib.seq2seq.BasicDecoder(cell_dec_attn, infer_helper, dec_start, output_layer = output_fn)
			infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_infer, impute_finished = True,
																 maximum_iterations=args.max_decoder_length, scope = "decoder_rnn")
			# [batch, max_decoder_len, vocab_size]
			self.decoder_distribution = infer_outputs.rnn_output
			# 这里在计算索引概率最大值的去除前面两个<pad>和<unk>的影响
			self.generation_index = tf.argmax(tf.split(self.decoder_distribution,
				[2, data.vocab_size-2], 2)[1], 2) + 2 # for removing UNK

		# calculate the gradient of parameters and update
		self.params = [k for k in tf.trainable_variables() if args.name in k.name]
		opt = tf.train.AdamOptimizer(self.learning_rate)          # 定义优化器
		gradients = tf.gradients(self.decoder_loss, self.params)  # 计算参数的梯度
		clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients,
				args.grad_clip)   # 梯度裁剪
		self.update = opt.apply_gradients(zip(clipped_gradients, self.params), 
				global_step=self.global_step)  # 对参数进行更新

		# save checkpoint
		self.latest_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
				max_to_keep=args.checkpoint_max_to_keep, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)
		self.best_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2,
				max_to_keep=1, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

		# create summary for tensorboard
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
		input_feed = {self.posts: data['post'], self.posts_length: data['post_length'], self.prevs_length: data['prev_length'],
				self.origin_responses: data['resp'], self.origin_responses_length: data['resp_length'],
					  self.is_train: True}
		if forward_only:
			output_feed = [self.decoder_loss, self.decoder_distribution_teacher]
		else:
			output_feed = [self.decoder_loss, self.gradient_norm, self.update]
		return session.run(output_feed, input_feed)

	def inference(self, session, data):
		input_feed = {self.posts: data['post'], self.posts_length: data['post_length'], self.prevs_length: data['prev_length'],
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
		# print(loss)
		return loss

	def train_process(self, sess, data, args):
		loss_step, time_step, epoch_step = np.zeros((1,)), .0, 0
		previous_losses = [1e18] * 3
		best_valid = 1e18
		data.restart("train", batch_size=args.batch_size, shuffle=True)
		batched_data = data.get_next_batch("train")

		for i in range(2):
			logger.info(f"post@{i}: {data.convert_ids_to_tokens(batched_data['post_allvocabs'][i].tolist(), trim=False)}")
			logger.info(f"resp@{i}: {data.convert_ids_to_tokens(batched_data['resp_allvocabs'][i].tolist(), trim=False)}")

		for epoch_step in range(args.epochs):
			while batched_data != None:
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
						sess.run(self.learning_rate_decay_op)
					if dev_loss < best_valid:
						best_valid = dev_loss
						self.store_checkpoint(sess, '%s/checkpoint_best/%s' % (args.model_dir, args.name), "best", args.name)

					previous_losses = previous_losses[1:] + [np.sum(loss_step)]
					loss_step, time_step = np.zeros((1,)), .0

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

		loss_record = []
		cnt = 0
		while batched_data != None:

			for key in batched_data:
				if isinstance(batched_data[key], np.ndarray):
					batched_data[key] = batched_data[key].tolist()

			batched_data['resp_length'] = [len(batched_data['resp'][0])]
			batched_data['resp'] = batched_data['resp']
			for each_resp in test_distractors[cnt]:
				batched_data['resp'].append([data.go_id] + data.convert_tokens_to_ids(jieba.lcut(each_resp)) + [data.eos_id])
				batched_data['resp_length'].append(len(batched_data['resp'][-1]))
			max_length = max(batched_data['resp_length'])
			resp = np.zeros((len(batched_data['resp']), max_length), dtype=int)
			for i, each_resp in enumerate(batched_data['resp']):
				resp[i, :len(each_resp)] = each_resp
			batched_data['resp'] = resp

			post = []
			post_length = []
			prev_length = []

			for _ in range(len(resp)):
				post += batched_data['post']
				post_length += batched_data['post_length']
				prev_length += batched_data['prev_length']

			batched_data['post'] = post
			batched_data['post_length'] = post_length
			batched_data['prev_length'] = prev_length

			_, _, loss = self.inference(sess, batched_data)
			loss_record.append(loss)
			cnt += 1
			batched_data = data.get_next_batch("test")

		assert cnt == len(test_distractors)

		loss = np.array(loss_record)
		loss_rank = np.argsort(loss, axis=1)
		hits1 = float(np.mean(loss_rank[:, 0] == 0))
		hits3 = float(np.mean(np.min(loss_rank[:, :3], axis=1) == 0))
		hits5 = float(np.mean(np.min(loss_rank[:, :5], axis=1) == 0))
		return {'hits@1' : hits1, 'hits@3': hits3, 'hits@5': hits5}

	def test_process(self, sess, data, args):
		metric1 = data.get_teacher_forcing_metric()  # 这里主要计算ppl指标
		metric2 = data.get_inference_metric()        # 这里主要计算bleu和distinct指标
		data.restart("test", batch_size=args.batch_size, shuffle=False)
		batched_data = data.get_next_batch("test")

		for i in range(3):
			logger.info(f"post@{i}: {data.convert_ids_to_tokens(batched_data['post_allvocabs'][i].tolist(), trim=False)}")
			logger.info(f"resp@{i}: {data.convert_ids_to_tokens(batched_data['resp_allvocabs'][i].tolist(), trim=False)}")

		while batched_data != None:
			batched_responses_id, gen_log_prob, _ = self.inference(sess, batched_data)
			metric1_data = {'resp_allvocabs': np.array(batched_data['resp_allvocabs']),
							'resp_length': np.array(batched_data['resp_length']),
							'gen_log_prob': np.array(gen_log_prob)}
			metric1.forward(metric1_data)
			batch_results = []
			for response_id in batched_responses_id:
				response_id_list = response_id.tolist()
				if data.eos_id in response_id_list:
					result_id = response_id_list[:response_id_list.index(data.eos_id)+1]
				else:
					result_id = response_id_list
				batch_results.append(result_id)

			metric2_data = {'gen': np.array(batch_results),
							'resp_allvocabs': np.array(batched_data['resp_allvocabs'])}
			metric2.forward(metric2_data)
			batched_data = data.get_next_batch("test")

		res = metric1.close()
		res.update(metric2.close())
		res.update(self.test_process_hits(sess, data, args))

		test_file = args.out_dir + "/%s_%s.txt" % (args.name, "test")
		with open(test_file, 'w', encoding="utf-8") as f:
			print("Test Result:")
			res_print = list(res.items())
			res_print.sort(key=lambda x: x[0])
			for key, value in res_print:
				if isinstance(value, float):
					print("\t%s:\t%f" % (key, value))
					f.write("%s:\t%f\n" % (key, value))
			f.write('\n')
			for i in range(len(res['resp'])):
				f.write("resp:\t%s\n" % " ".join(res['resp'][i]))
				f.write("gen:\t%s\n\n" % " ".join(res['gen'][i]))

		logger.info("result output to %s." % test_file)
		return {key: val for key, val in res.items() if type(val) in [bytes, int, float]}
