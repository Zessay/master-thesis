# coding=utf-8
# @Author: 莫冉
# @Date: 2021-01-08
import numpy as np
import tensorflow as tf
import time
import logging

from utils.output_projection import output_projection_layer, MyDense, MyInferenceHelper, MyAttention
from utils import SummaryHelper

import os
import jieba
import json

logger = logging.getLogger(__file__)


class HredModel(object):
	def __init__(self, data, args, embed):
		# self.init_states = tf.placeholder(tf.float32, (None, args.ch_size), 'ctx_inps')  # batch*ch_size
		# 以下各个变量同上面的hred模型
		self.posts = tf.placeholder(tf.int32, (None, None), 'enc_inps')
		self.posts_length = tf.placeholder(tf.int32, (None,), 'enc_lens')
		self.prev_posts = tf.placeholder(tf.int32, (None, None), 'enc_prev_inps')
		self.prev_posts_length = tf.placeholder(tf.int32, (None,), 'enc_prev_lens')

		# kgs: [batch, max_kg_nums, max_kg_length]
		# kgs_h_length: [batch, max_kg_nums]
		# kgs_hr_length: [batch, max_kg_nums]
		# kgs_hrt_length: [batch, max_kg_nums]
		# kgs_index: [batch, max_kg_nums]
		self.kgs = tf.placeholder(tf.int32, (None, None, None), 'kg_inps')
		self.kgs_h_length = tf.placeholder(tf.int32, (None, None), 'kg_h_lens')
		self.kgs_hr_length = tf.placeholder(tf.int32, (None, None), 'kg_hr_lens')
		self.kgs_hrt_length = tf.placeholder(tf.int32, (None, None), 'kg_hrt_lens')
		self.kgs_index = tf.placeholder(tf.float32, (None, None), 'kg_indices')

		self.origin_responses = tf.placeholder(tf.int32, (None, None), 'dec_inps')  # batch*len
		self.origin_responses_length = tf.placeholder(tf.int32, (None,), 'dec_lens')  # batch
		self.context_length = tf.placeholder(tf.int32, (None,), 'ctx_lens')

		# 用来平衡解码损失和kg损失的超参数
		self.lamb = tf.placeholder(tf.float32, name="lamb")
		self.is_train = tf.placeholder(tf.bool)

		# 这里表示当前batch中最大的轮次
		num_past_turns = tf.shape(self.posts)[0] // tf.shape(self.origin_responses)[0]

		# deal with original data to adapt encoder and decoder
		# 获取解码器的输入和输出
		batch_size, decoder_len = tf.shape(self.origin_responses)[0], tf.shape(self.origin_responses)[1]
		self.responses = tf.split(self.origin_responses, [1, decoder_len-1], 1)[1] # no go_id
		self.responses_length = self.origin_responses_length - 1
		self.responses_input = tf.split(self.origin_responses, [decoder_len-1, 1], 1)[0] # no eos_id
		self.responses_target = self.responses
		decoder_len = decoder_len - 1
		# 编码器的输入
		self.posts_input = self.posts
		# 解码器的mask矩阵
		self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length-1,
			decoder_len), reverse=True, axis=1), [-1, decoder_len])
		# key表示知识中h,r的，所以key_mask即对应h,r位置为1
		# value表示知识中的t，所以value_mask即对应t位置为1
		kg_len = tf.shape(self.kgs)[2]
		kg_h_mask = tf.reshape(tf.cumsum(tf.one_hot(self.kgs_h_length-1,
			kg_len), reverse=True, axis=2), [batch_size, -1, kg_len, 1])
		kg_hr_mask = tf.reshape(tf.cumsum(tf.one_hot(self.kgs_hr_length-1,
			kg_len), reverse=True, axis=2), [batch_size, -1, kg_len, 1])
		kg_hrt_mask = tf.reshape(tf.cumsum(tf.one_hot(self.kgs_hrt_length-1,
			kg_len), reverse=True, axis=2), [batch_size, -1, kg_len, 1])
		kg_key_mask = kg_hr_mask                    # [batch, max_kg_nums, max_kg_len, 1]
		kg_value_mask = kg_hrt_mask - kg_hr_mask    # [batch, max_kg_nums, max_kg_len, 1]

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

		# 将编码器输入、解码器输入以及kg输入转换为向量
		self.encoder_input = tf.nn.embedding_lookup(self.embed, self.posts)
		self.decoder_input = tf.nn.embedding_lookup(self.embed, self.responses_input)
		self.kg_input = tf.nn.embedding_lookup(self.embed, self.kgs)
		#self.knowledge_max = tf.reduce_max(tf.where(tf.cast(tf.tile(knowledge_mask, [1, 1, args.embedding_size]), tf.bool), self.knowledge_input, -mask_value), axis=1)
		#self.knowledge_min = tf.reduce_max(tf.where(tf.cast(tf.tile(knowledge_mask, [1, 1, args.embedding_size]), tf.bool), self.knowledge_input, mask_value), axis=1)
		# 得到每一个知识key, value的向量表征，即词向量的平均
		# [batch, max_kg_nums, embed_size]
		self.kg_key_avg = tf.reduce_sum(self.kg_input * kg_key_mask, axis=2) / tf.maximum(tf.reduce_sum(kg_key_mask, axis=2), tf.ones_like(tf.expand_dims(self.kgs_hrt_length, -1), dtype=tf.float32))
		self.kg_value_avg = tf.reduce_sum(self.kg_input * kg_value_mask, axis=2) / tf.maximum(tf.reduce_sum(kg_value_mask, axis=2), tf.ones_like(tf.expand_dims(self.kgs_hrt_length, -1), dtype=tf.float32))

		#self.encoder_input = tf.cond(self.is_train,
		#							 lambda: tf.nn.dropout(tf.nn.embedding_lookup(self.embed, self.posts_input), 0.8),
		#							 lambda: tf.nn.embedding_lookup(self.embed, self.posts_input))  # batch*len*unit
		#self.decoder_input = tf.cond(self.is_train,
		#							 lambda: tf.nn.dropout(tf.nn.embedding_lookup(self.embed, self.responses_input), 0.8),
		#							 lambda: tf.nn.embedding_lookup(self.embed, self.responses_input))

		# build rnn_cell
		cell_enc = tf.nn.rnn_cell.GRUCell(args.eh_size)
		cell_ctx = tf.nn.rnn_cell.GRUCell(args.ch_size)
		cell_dec = tf.nn.rnn_cell.GRUCell(args.dh_size)

		# build encoder
		with tf.variable_scope('encoder'):
			# 对每一句话编码得到每一个语句的表征
			# encoder_output: [batch*(num_turns-1), max_post_length, eh_size]
			# encoder_state: [batch*(num_turns-1), eh_size]
			encoder_output, encoder_state = tf.nn.dynamic_rnn(cell_enc, self.encoder_input,
				self.posts_length, dtype=tf.float32, scope="encoder_rnn")

		with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
			# 对于post中的最后一句进行编码
			# prev_output: [batch, max_prev_length, eh_size]
			prev_output, _ = tf.nn.dynamic_rnn(cell_enc, tf.nn.embedding_lookup(self.embed, self.prev_posts), self.prev_posts_length,
										 dtype=tf.float32, scope="encoder_rnn")
		

		with tf.variable_scope('context'):
			# 对语句进行编码，得到整个上下文的表征
			# context_sate: [batch, ch_size]
			encoder_state_reshape = tf.reshape(encoder_state, [-1, num_past_turns, args.eh_size])
			_, self.context_state = tf.nn.dynamic_rnn(cell_ctx, encoder_state_reshape,
												self.context_length, dtype=tf.float32, scope='context_rnn')

		# get output projection function
		output_fn = MyDense(data.vocab_size, use_bias = True)
		sampled_sequence_loss = output_projection_layer(args.dh_size, data.vocab_size, args.softmax_samples)

		# construct attention
		'''
		encoder_len = tf.shape(encoder_output)[1]
		attention_memory = tf.reshape(encoder_output, [batch_size, -1, args.eh_size])
		attention_mask = tf.reshape(tf.sequence_mask(self.posts_length, encoder_len), [batch_size, -1])
		attention_mask = tf.concat([tf.ones([batch_size, 1], tf.bool), attention_mask[:,1:]], axis=1)
		attn_mechanism = MyAttention(args.dh_size, attention_memory, attention_mask)
		'''
		# 定义注意力计算方式
		# 这里输入prev_output只有最近一轮，所以直接使用BahdanauAttention，不需要使用MyAttention
		attn_mechanism = tf.contrib.seq2seq.BahdanauAttention(args.dh_size, prev_output,
				memory_sequence_length=tf.maximum(self.prev_posts_length, 1))
		cell_dec_attn = tf.contrib.seq2seq.AttentionWrapper(cell_dec, attn_mechanism,
				attention_layer_size=args.dh_size)
		# 将上下文编码映射为解码器编码，作为初始状态 [batch, dh_size]
		ctx_state_shaping = tf.layers.dense(self.context_state, args.dh_size, activation=None)
		dec_start = cell_dec_attn.zero_state(batch_size, dtype=tf.float32).clone(cell_state=ctx_state_shaping)

		# calculate kg embedding
		with tf.variable_scope('knowledge'):
			query = tf.reshape(tf.layers.dense(tf.concat(self.context_state, axis=-1), args.embedding_size, use_bias=False), [batch_size, 1, args.embedding_size])
		kg_score = tf.reduce_sum(query * self.kg_key_avg, axis=2)
		kg_score = tf.where(tf.greater(self.kgs_hrt_length, 0), kg_score, - tf.ones_like(kg_score) * np.inf)
		# [batch, max_kg_nums]
		kg_alignment = tf.nn.softmax(kg_score)
		kg_max = tf.argmax(kg_alignment, axis=-1)
		kg_max_onehot = tf.one_hot(kg_max, tf.shape(kg_alignment)[1], dtype=tf.float32)
		self.kg_acc = tf.reduce_sum(kg_max_onehot * self.kgs_index) / tf.maximum(tf.reduce_sum(tf.reduce_max(self.kgs_index, axis=-1)), tf.constant(1.0))
		self.kg_loss = tf.reduce_sum(- tf.log(tf.clip_by_value(kg_alignment, 1e-12, 1.0)) * self.kgs_index, axis=1) / tf.maximum(tf.reduce_sum(self.kgs_index, axis=1), tf.ones([batch_size], dtype=tf.float32))
		self.kg_loss = tf.reduce_mean(self.kg_loss)
		# # 计算kg的mask矩阵 [batch, max_kg_nums, 1]，有知识的地方对应为1，没有的地方对应为0
		# kg_num_mask = tf.expand_dims(tf.greater(self.kgs_hrt_length, 0), axis=-1)

		#
		self.knowledge_embed = tf.reduce_sum(tf.expand_dims(kg_alignment, axis=-1) * self.kg_value_avg, axis=1)
		#self.knowledge_embed = tf.Print(self.knowledge_embed, ['acc', self.kg_acc, 'loss', self.kg_loss])
		knowledge_embed_extend = tf.tile(tf.expand_dims(self.knowledge_embed, axis=1), [1, decoder_len, 1])
		self.decoder_input = tf.concat([self.decoder_input, knowledge_embed_extend], axis=2)
		# construct helper
		train_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_input, tf.maximum(self.responses_length, 1))
		infer_helper = MyInferenceHelper(self.embed, tf.fill([batch_size], data.go_id), data.eos_id, self.knowledge_embed)
		#infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embed, tf.fill([batch_size], data.go_id), data.eos_id)

		# build decoder (train)
		with tf.variable_scope('decoder'):
			decoder_train = tf.contrib.seq2seq.BasicDecoder(cell_dec_attn, train_helper, dec_start)
			train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_train, impute_finished = True, scope = "decoder_rnn")
			# decoder_output: [batch, max_resp_length, dh_size]
			self.decoder_output = train_outputs.rnn_output
			#self.decoder_output = tf.nn.dropout(self.decoder_output, 0.8)
			# decoder_distribution_teacher: [batch, max_resp_length, vocab_size]
			# decoder_loss: 标量
			# decoder_all_loss: [batch]，表示每一句的损失
			self.decoder_distribution_teacher, self.decoder_loss, self.decoder_all_loss = \
				sampled_sequence_loss(self.decoder_output, self.responses_target, self.decoder_mask)

		# build decoder (test)
		with tf.variable_scope('decoder', reuse=True):
			decoder_infer = tf.contrib.seq2seq.BasicDecoder(cell_dec_attn, infer_helper, dec_start, output_layer = output_fn)
			infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_infer, impute_finished = True,
					maximum_iterations=args.max_decoder_length, scope = "decoder_rnn")
			# decoder_distribution: [batch, max_decoder_length, vocab_size]
			# generation_index: [batch, max_decoder_length]
			self.decoder_distribution = infer_outputs.rnn_output
			self.generation_index = tf.argmax(tf.split(self.decoder_distribution,
				[2, data.vocab_size-2], 2)[1], 2) + 2 # for removing UNK

		# calculate the gradient of parameters and update
		self.params = [k for k in tf.trainable_variables() if args.name in k.name]
		opt = tf.train.AdamOptimizer(self.learning_rate)
		self.loss = self.decoder_loss + self.lamb * self.kg_loss
		gradients = tf.gradients(self.loss, self.params)
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
		self.create_summary(args)

	def store_checkpoint(self, sess, path, key, name):
		if key == "latest":
			self.latest_saver.save(sess, path, global_step=self.global_step, latest_filename=name)
		else:
			self.best_saver.save(sess, path, global_step=self.global_step, latest_filename=name)


	def create_summary(self, args):
		self.summaryHelper = SummaryHelper("%s/%s_%s" % \
				(args.log_dir, args.name, time.strftime("%H%M%S", time.localtime())), args)

		self.trainSummary = self.summaryHelper.addGroup(scalar=["loss", "perplexity"], prefix="train")

		scalarlist = ["loss", "perplexity"]
		tensorlist = []
		textlist = []
		emblist = []
		for i in args.show_sample:
			textlist.append("show_str%d" % i)
		self.devSummary = self.summaryHelper.addGroup(scalar=scalarlist, tensor=tensorlist, text=textlist,
				embedding=emblist, prefix="dev")
		self.testSummary = self.summaryHelper.addGroup(scalar=scalarlist, tensor=tensorlist, text=textlist,
				embedding=emblist, prefix="test")


	def print_parameters(self):
		for item in self.params:
			logger.info('%s: %s' % (item.name, item.get_shape()))

	def step_decoder(self, sess, data, forward_only=False, inference=False, lamb=1.0):
		input_feed = {
				#self.init_states: data['init_states'],
				self.posts: data['posts'],
				self.posts_length: data['posts_length'],
				self.prev_posts: data['prev_posts'],
				self.prev_posts_length: data['prev_posts_length'],
				self.origin_responses: data['responses'],
				self.origin_responses_length: data['responses_length'],
				self.context_length: data['context_length'],
				self.kgs: data['kg'],
				self.kgs_h_length: data['kg_h_length'],
				self.kgs_hr_length: data['kg_hr_length'],
				self.kgs_hrt_length: data['kg_hrt_length'],
				self.kgs_index: data['kg_index'],
				self.lamb: lamb
				}

		if inference:
			input_feed.update({self.is_train: False})
			output_feed = [self.generation_index, self.decoder_distribution_teacher, self.decoder_all_loss,
						   self.kg_loss, self.kg_acc]
		else:
			input_feed.update({self.is_train: True})
			if forward_only:
				output_feed = [self.decoder_loss, self.decoder_distribution_teacher, self.kg_loss, self.kg_acc]
			else:
				output_feed = [self.decoder_loss, self.gradient_norm, self.update, self.kg_loss, self.kg_acc]

		return sess.run(output_feed, input_feed)

	def evaluate(self, sess, data, batch_size, key_name, lamb=1.0):
		loss = np.zeros((3,))
		total_length = np.zeros((3,))
		data.restart(key_name, batch_size=batch_size, shuffle=False)
		batched_data = data.get_next_batch(key_name)
		while batched_data != None:
			decoder_loss, _, kg_loss, kg_acc = self.step_decoder(sess, batched_data, forward_only=True, lamb=lamb)
			length = np.sum(np.maximum(np.array(batched_data['responses_length']) - 1, 0))
			kg_length = np.sum(np.max(batched_data['kg_index'], axis=-1))
			total_length += [length, kg_length, kg_length]
			loss += [decoder_loss * length, kg_loss * kg_length, kg_acc * kg_length]
			batched_data = data.get_next_batch(key_name)
		loss /= total_length
		logger.info('perplexity on %s set: %.2f, kg_ppx: %.2f, kg_loss: %.4f, kg_acc: %.4f' % (key_name, np.exp(loss[0]), np.exp(loss[1]), loss[1], loss[2]))
		return loss

	def train_process(self, sess, data, args):
		loss_step, time_step, epoch_step = np.zeros((3,)), .0, 0
		previous_losses = [1e18] * 3
		best_valid = 1e18
		data.restart("train", batch_size=args.batch_size, shuffle=True)
		batched_data = data.get_next_batch("train")

		for epoch_step in range(args.epochs):
			while batched_data != None:
				if self.global_step.eval() % args.checkpoint_steps == 0 and self.global_step.eval() != 0:
					logger.info(
						"Epoch %d global step %d learning rate %.4f step-time %.2f perplexity: %.2f, kg_ppx: %.2f, kg_loss: %.4f, kg_acc: %.4f" % (
						epoch_step, self.global_step.eval(), self.learning_rate.eval(), time_step, np.exp(loss_step[0]),
						np.exp(loss_step[1]), loss_step[1], loss_step[2]))
					self.trainSummary(self.global_step.eval() // args.checkpoint_steps,
									  {'loss': loss_step[0], 'perplexity': np.exp(loss_step[0])})
					self.store_checkpoint(sess, '%s/checkpoint_latest/%s' % (args.model_dir, args.name), "latest",
										  args.name)

					dev_loss = self.evaluate(sess, data, args.batch_size, "dev", lamb=args.lamb)
					self.devSummary(self.global_step.eval() // args.checkpoint_steps,
									{'loss': dev_loss[0], 'perplexity': np.exp(dev_loss[0])})

					if np.sum(loss_step) > max(previous_losses):
						sess.run(self.learning_rate_decay_op)
					if dev_loss[0] < best_valid:
						best_valid = dev_loss[0]
						self.store_checkpoint(sess, '%s/checkpoint_best/%s' % (args.model_dir, args.name), "best",
											  args.name)

					previous_losses = previous_losses[1:] + [np.sum(loss_step[0])]
					loss_step, time_step = np.zeros((3,)), .0

				start_time = time.time()
				step_out = self.step_decoder(sess, batched_data, lamb=args.lamb)
				loss_step += np.array([step_out[0], step_out[3], step_out[4]]) / args.checkpoint_steps
				time_step += (time.time() - start_time) / args.checkpoint_steps
				batched_data = data.get_next_batch("train")

			data.restart("train", batch_size=args.batch_size, shuffle=True)
			batched_data = data.get_next_batch("train")


	def test_process_hits(self, sess, data, args):

		with open(os.path.join(args.datapath, 'test_distractors.json'), 'r', encoding='utf8') as f:
			test_distractors = json.load(f)

		data.restart("test", batch_size=1, shuffle=False)
		batched_data = data.get_next_batch("test")

		loss_record = []
		cnt = 0
		while batched_data != None:

			for key in batched_data:
				if isinstance(batched_data[key], np.ndarray):
					batched_data[key] = batched_data[key].tolist()

			batched_data['responses_length'] = [len(batched_data['responses'][0])]
			for each_resp in test_distractors[cnt]:
				batched_data['responses'].append([data.go_id] + data.convert_tokens_to_ids(jieba.lcut(each_resp)) +
												 [data.eos_id])
				batched_data['responses_length'].append(len(batched_data['responses'][-1]))
			max_length = max(batched_data['responses_length'])
			resp = np.zeros((len(batched_data['responses']), max_length), dtype=int)
			for i, each_resp in enumerate(batched_data['responses']):
				resp[i, :len(each_resp)] = each_resp
			batched_data['responses'] = resp

			posts = []
			posts_length = []
			prev_posts = []
			prev_posts_length = []
			context_length = []

			kg = []
			kg_h_length = []
			kg_hr_length = []
			kg_hrt_length = []
			kg_index = []

			for _ in range(len(resp)):
				posts += batched_data['posts']
				posts_length += batched_data['posts_length']
				prev_posts += batched_data['prev_posts']
				prev_posts_length += batched_data['prev_posts_length']
				context_length += batched_data['context_length']

				kg += batched_data['kg']
				kg_h_length += batched_data['kg_h_length']
				kg_hr_length += batched_data['kg_hr_length']
				kg_hrt_length += batched_data['kg_hrt_length']
				kg_index += batched_data['kg_index']

			batched_data['posts'] = posts
			batched_data['posts_length'] = posts_length
			batched_data['prev_posts'] = prev_posts
			batched_data['prev_posts_length'] = prev_posts_length
			batched_data['context_length'] = context_length

			batched_data['kg'] = kg
			batched_data['kg_h_length'] = kg_h_length
			batched_data['kg_hr_length'] = kg_hr_length
			batched_data['kg_hrt_length'] = kg_hrt_length
			batched_data['kg_index'] = kg_index

			_, _, loss, _, _ = self.step_decoder(sess, batched_data, inference=True, lamb=args.lamb)
			loss_record.append(loss)
			cnt += 1

			batched_data = data.get_next_batch("test")

		assert cnt == len(test_distractors)

		loss = np.array(loss_record)
		loss_rank = np.argsort(loss, axis=1)
		hits1 = float(np.mean(loss_rank[:, 0] == 0))
		hits3 = float(np.mean(np.min(loss_rank[:, :3], axis=1) == 0))
		return {'hits@1' : hits1, 'hits@3': hits3}

	def test_process(self, sess, data, args):

		metric1 = data.get_teacher_forcing_metric()
		metric2 = data.get_inference_metric()
		data.restart("test", batch_size=args.batch_size, shuffle=False)
		batched_data = data.get_next_batch("test")

		while batched_data != None:
			batched_responses_id, gen_log_prob, _, _, _ = self.step_decoder(sess, batched_data, inference=True, lamb=args.lamb)
			metric1_data = {'resp_allvocabs': np.array(batched_data['responses_allvocabs']),
							'resp_length': np.array(batched_data['responses_length']),
							'gen_log_prob': np.array(gen_log_prob)}
			metric1.forward(metric1_data)
			batch_results = []
			for response_id in batched_responses_id:
				response_id_list = response_id.tolist()
				if data.eos_id in response_id_list:
					result_id = response_id_list[:response_id_list.index(data.eos_id) + 1]
				else:
					result_id = response_id_list
				batch_results.append(result_id)

			metric2_data = {'gen': np.array(batch_results),
							'resp_allvocabs': np.array(batched_data['responses_allvocabs'])}
			metric2.forward(metric2_data)
			batched_data = data.get_next_batch("test")

		res = metric1.close()
		res.update(metric2.close())
		res.update(self.test_process_hits(sess, data, args))

		test_file = args.output_dir + "/%s_%s.txt" % (args.name, "test")
		with open(test_file, 'w') as f:
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

		logger.info("result output to %s" % test_file)
		return {key: val for key, val in res.items() if type(val) in [bytes, int, float]}
