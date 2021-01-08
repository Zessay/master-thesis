# coding=utf-8
# @Author: 莫冉
# @Date: 2021-01-08
import numpy as np
import tensorflow as tf
import time

import logging
from tensorflow.python.ops.nn import dynamic_rnn
from utils.output_projection import output_projection_layer, MyDense, MyAttention, MyInferenceHelper
from utils import SummaryHelper
import jieba
import os
import json

logger = logging.getLogger(__file__)


class Seq2SeqModel(object):
	def __init__(self, data, args, embed):
		# 这里的输入和前面的seq2seq一致
		self.posts = tf.placeholder(tf.int32, (None, None), 'enc_inps')  # batch*len
		self.posts_length = tf.placeholder(tf.int32, (None,), 'enc_lens')  # batch
		self.prevs_length = tf.placeholder(tf.int32, (None,), 'enc_lens_prev')  # batch
		self.origin_responses = tf.placeholder(tf.int32, (None, None), 'dec_inps')  # batch*len
		self.origin_responses_length = tf.placeholder(tf.int32, (None,), 'dec_lens')  # batch

		# kgs表示该样例所在这段对话中所有的知识：[batch, max_kg_nums, max_kg_length]
		# kgs_h_length表示每一个知识中head entity的长度：[batch, max_kg_nums]
		# kgs_hr_length表示每一个知识中head entity和relation的长度：[batch, max_kg_nums]
		# kgs_hrt_length表示每一个知识中h,r,t的长度：[batch, max_kg_nums]
		# kgs_index表示当前这句话实际使用的kg的索引指示矩阵：[batch, max_kg_nums]（其中使用的知识对应为1，没有使用的知识对应为0）
		self.kgs = tf.placeholder(tf.int32, (None, None, None), 'kg_inps')
		self.kgs_h_length = tf.placeholder(tf.int32, (None, None), 'kg_h_lens')
		self.kgs_hr_length = tf.placeholder(tf.int32, (None, None), 'kg_hr_lens')
		self.kgs_hrt_length = tf.placeholder(tf.int32, (None, None), 'kg_hrt_lens')
		self.kgs_index = tf.placeholder(tf.float32, (None, None), 'kg_indices')

		# 用来平衡解码损失和kg损失的超参数
		self.lamb = tf.placeholder(tf.float32, name='lamb')
		self.is_train = tf.placeholder(tf.bool)

		# deal with original data to adapt encoder and decoder
		# 获取解码器的输入和输出
		batch_size, decoder_len = tf.shape(self.origin_responses)[0], tf.shape(self.origin_responses)[1]
		self.responses = tf.split(self.origin_responses, [1, decoder_len-1], 1)[1] # no go_id
		self.responses_length = self.origin_responses_length - 1
		self.responses_input = tf.split(self.origin_responses, [decoder_len-1, 1], 1)[0] # no eos_id
		self.responses_target = self.responses
		decoder_len = decoder_len - 1
		# 获取编码器的输入
		self.posts_input = self.posts   # batch*len
		# 对解码器的mask矩阵，对于pad的mask
		self.decoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.responses_length-1,
			decoder_len), reverse=True, axis=1), [-1, decoder_len])
		kg_len = tf.shape(self.kgs)[2]
		#kg_len = tf.Print(kg_len, [batch_size, kg_len, decoder_len, self.kgs_length])
		# kg_h_mask = tf.reshape(tf.cumsum(tf.one_hot(self.kgs_h_length-1,
		# 	kg_len), reverse=True, axis=2), [batch_size, -1, kg_len, 1])
		# 这里分别得到对于key（也就是hr）的mask矩阵：[batch_size, max_kg_nums, max_kg_length, 1]
		# 以及对于value（也就是t）的mask矩阵：[batch_size, max_kg_nums, max_kg_length, 1]
		kg_hr_mask = tf.reshape(tf.cumsum(tf.one_hot(self.kgs_hr_length-1,
			kg_len), reverse=True, axis=2), [batch_size, -1, kg_len, 1])
		kg_hrt_mask = tf.reshape(tf.cumsum(tf.one_hot(self.kgs_hrt_length-1,
			kg_len), reverse=True, axis=2), [batch_size, -1, kg_len, 1])
		kg_key_mask = kg_hr_mask
		kg_value_mask = kg_hrt_mask - kg_hr_mask

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
		# encoder_input: [batch, encoder_len, embed_size]
		self.encoder_input = tf.nn.embedding_lookup(self.embed, self.posts)
		# decoder_input: [batch, decoder_len, embed_size]
		self.decoder_input = tf.nn.embedding_lookup(self.embed, self.responses_input)
		# kg_input: [batch, max_kg_nums, max_kg_length, embed_size]
		self.kg_input = tf.nn.embedding_lookup(self.embed, self.kgs)
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
			encoder_output, encoder_state = dynamic_rnn(cell_enc, self.encoder_input,
				self.posts_length, dtype=tf.float32, scope="encoder_rnn")
		# key对应一个知识h,r的词向量的均值 [batch, max_kg_nums, embed_size]
		# value对应一个知识t的词向量的均值 [batch, max_kg_nums, embed_size]
		self.kg_key_avg = tf.reduce_sum(self.kg_input * kg_key_mask, axis=2) / tf.maximum(tf.reduce_sum(kg_key_mask, axis=2), tf.ones_like(tf.expand_dims(self.kgs_hrt_length, -1), dtype=tf.float32))
		self.kg_value_avg = tf.reduce_sum(self.kg_input * kg_value_mask, axis=2) / tf.maximum(tf.reduce_sum(kg_value_mask, axis=2), tf.ones_like(tf.expand_dims(self.kgs_hrt_length, -1), dtype=tf.float32))
		# 将编码器的输出状态映射到embed_size的维度
		# query: [batch, 1, embed_size]
		with tf.variable_scope('knowledge'):
			query = tf.reshape(tf.layers.dense(tf.concat(encoder_state, axis=-1), args.embedding_size, use_bias=False), [batch_size, 1, args.embedding_size])
		# [batch, max_kg_nums]
		kg_score = tf.reduce_sum(query * self.kg_key_avg, axis=2)
		# 对于hrt大于0的位置（即该位置存在知识），取对应的kg_score，否则对应位置为-inf
		kg_score = tf.where(tf.greater(self.kgs_hrt_length, 0), kg_score, - tf.ones_like(kg_score) * np.inf)
		# 计算每个知识对应的分数 [batch, max_kg_nums]
		kg_alignment = tf.nn.softmax(kg_score)

		# 根据计算的kg注意力分数的位置，计算关注的kg准确率和损失
		kg_max = tf.argmax(kg_alignment, axis=-1)
		kg_max_onehot = tf.one_hot(kg_max, tf.shape(kg_alignment)[1], dtype=tf.float32)
		self.kg_acc = tf.reduce_sum(kg_max_onehot * self.kgs_index) / tf.maximum(
			tf.reduce_sum(tf.reduce_max(self.kgs_index, axis=-1)), tf.constant(1.0))
		self.kg_loss = tf.reduce_sum(- tf.log(tf.clip_by_value(kg_alignment, 1e-12, 1.0)) * self.kgs_index, axis=1) / tf.maximum(
			tf.reduce_sum(self.kgs_index, axis=1), tf.ones([batch_size], dtype=tf.float32))
		self.kg_loss = tf.reduce_mean(self.kg_loss)
		# 得到注意力之后的知识的嵌入：[batch, embed_size]
		self.knowledge_embed = tf.reduce_sum(tf.expand_dims(kg_alignment, axis=-1) * self.kg_value_avg, axis=1)
		# 对维度进行扩充[batch, decoder_len, embed_size]
		knowledge_embed_extend = tf.tile(tf.expand_dims(self.knowledge_embed, axis=1), [1, decoder_len, 1])
		# 将知识和原始的解码输入拼接，作为新的解码输入 [batch, decoder_len, 2*embed_size]
		self.decoder_input = tf.concat([self.decoder_input, knowledge_embed_extend], axis=2)

		# get output projection function
		output_fn = MyDense(data.vocab_size, use_bias = True)
		sampled_sequence_loss = output_projection_layer(args.dh_size, data.vocab_size, args.softmax_samples)

		encoder_len = tf.shape(encoder_output)[1]
		posts_mask = tf.sequence_mask(self.posts_length, encoder_len)
		prevs_mask = tf.sequence_mask(self.prevs_length, encoder_len)
		attention_mask = tf.reshape(tf.logical_xor(posts_mask, prevs_mask), [batch_size, encoder_len])

		# construct helper and attention
		train_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_input, self.responses_length)
		#infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embed, tf.fill([batch_size], data.go_id), data.eos_id)
		# 为了在推理的时候，每一次的输入都是上一次输出和知识的拼接
		infer_helper = MyInferenceHelper(self.embed, tf.fill([batch_size], data.go_id), data.eos_id, self.knowledge_embed)
		#attn_mechanism = tf.contrib.seq2seq.BahdanauAttention(args.dh_size, encoder_output,
														 #  memory_sequence_length=self.posts_length)
		# 这里的MyAttention主要解决BahdanauAttention只能输入编码序列长度的问题
		attn_mechanism = MyAttention(args.dh_size, encoder_output, attention_mask)
		cell_dec_attn = tf.contrib.seq2seq.AttentionWrapper(cell_dec, attn_mechanism,
															attention_layer_size=args.dh_size)
		enc_state_shaping = tf.layers.dense(encoder_state, args.dh_size, activation = None)
		dec_start = cell_dec_attn.zero_state(batch_size, dtype = tf.float32).clone(cell_state = enc_state_shaping)

		# build decoder (train)
		with tf.variable_scope('decoder'):
			decoder_train = tf.contrib.seq2seq.BasicDecoder(cell_dec_attn, train_helper, dec_start)
			train_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_train, impute_finished = True, scope = "decoder_rnn")
			self.decoder_output = train_outputs.rnn_output
			#self.decoder_output = tf.nn.dropout(self.decoder_output, 0.8)
			# 输出概率分布和解码损失
			self.decoder_distribution_teacher, self.decoder_loss, self.decoder_all_loss = \
				sampled_sequence_loss(self.decoder_output, self.responses_target, self.decoder_mask)

		# build decoder (test)
		with tf.variable_scope('decoder', reuse=True):
			decoder_infer = tf.contrib.seq2seq.BasicDecoder(cell_dec_attn, infer_helper, dec_start, output_layer = output_fn)
			infer_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder_infer, impute_finished = True,
																 maximum_iterations=args.max_decoder_length, scope = "decoder_rnn")
			# 输出解码概率分布
			self.decoder_distribution = infer_outputs.rnn_output
			self.generation_index = tf.argmax(tf.split(self.decoder_distribution,
				[2, data.vocab_size-2], 2)[1], 2) + 2 # for removing UNK

		# calculate the gradient of parameters and update
		self.params = [k for k in tf.trainable_variables() if args.name in k.name]
		opt = tf.train.AdamOptimizer(self.learning_rate)
		# 将解码损失和kg损失相加
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
		self.testSummary = self.summaryHelper.addGroup(scalar=scalarlist, tensor=tensorlist, text=textlist,
													  prefix="test")


	def print_parameters(self):
		for item in self.params:
			logger.info('%s: %s' % (item.name, item.get_shape()))
	
	def step_decoder(self, session, data, lamb=1.0, forward_only=False):
		input_feed = {
				self.posts: data['post'], 
				self.posts_length: data['post_length'], 
				self.prevs_length: data['prev_length'],
				self.origin_responses: data['resp'], 
				self.origin_responses_length: data['resp_length'],
				self.kgs: data['kg'],
				self.kgs_h_length: data['kg_h_length'],
				self.kgs_hr_length: data['kg_hr_length'],
				self.kgs_hrt_length: data['kg_hrt_length'],
				self.kgs_index: data['kg_index'],
				self.lamb: lamb,
				self.is_train: True
				}
		if forward_only:
			output_feed = [self.decoder_loss, self.decoder_distribution_teacher, self.kg_loss, self.kg_acc]
		else:
			output_feed = [self.decoder_loss, self.gradient_norm, self.update, self.kg_loss, self.kg_acc]
		return session.run(output_feed, input_feed)

	def inference(self, session, data, lamb=1.0):
		input_feed = {
				self.posts: data['post'], 
				self.posts_length: data['post_length'], 
				self.prevs_length: data['prev_length'],
				self.origin_responses: data['resp'], 
				self.origin_responses_length: data['resp_length'],
				self.kgs: data['kg'],
				self.kgs_h_length: data['kg_h_length'],
				self.kgs_hr_length: data['kg_hr_length'],
				self.kgs_hrt_length: data['kg_hrt_length'],
				self.kgs_index: data['kg_index'],
				self.lamb: lamb,
				self.is_train: False
				}
		output_feed = [self.generation_index, self.decoder_distribution_teacher,
					   self.decoder_all_loss, self.kg_loss, self.kg_acc]
		return session.run(output_feed, input_feed)

	def evaluate(self, sess, data, batch_size, key_name, lamb=1.0):
		loss = np.zeros((3,))
		total_length = np.zeros((3,))
		data.restart(key_name, batch_size=batch_size, shuffle=False)
		batched_data = data.get_next_batch(key_name)
		while batched_data != None:
			decoder_loss, _, kg_loss, kg_acc = self.step_decoder(sess, batched_data, lamb=lamb, forward_only=True)
			length = np.sum(np.maximum(np.array(batched_data['resp_length']) - 1, 0))
			kg_length = np.sum(np.max(batched_data['kg_index'], axis=-1))
			total_length += [length, kg_length, kg_length]
			loss += [decoder_loss * length, kg_loss * kg_length, kg_acc * kg_length]
			batched_data = data.get_next_batch(key_name)

		loss /= total_length
		logger.info('	perplexity on %s set: %.2f, kg_ppx: %.2f, kg_loss: %.4f, kg_acc: %.4f' % (
		key_name, np.exp(loss[0]), np.exp(loss[1]), loss[1], loss[2]))
		return loss

	def train_process(self, sess, data, args):
		loss_step, time_step, epoch_step = np.zeros((3,)), .0, 0
		previous_losses = [1e18] * 3
		best_valid = 1e18
		data.restart("train", batch_size=args.batch_size, shuffle=True)
		batched_data = data.get_next_batch("train")

		for i in range(2):
			logger.info(f"post@ {data.convert_ids_to_tokens(batched_data['post_allvocabs'][i].tolist(), trim=False)}")
			logger.info(f"length@ {batched_data['prev_length'][i], batched_data['post_length'][i]}")
			logger.info(f"last@ {data.convert_ids_to_tokens(batched_data['post_allvocabs'][i].tolist()[batched_data['prev_length'][i]: batched_data['post_length'][i]], trim=False)}")
			logger.info(f"resp@ {data.convert_ids_to_tokens(batched_data['resp_allvocabs'][i].tolist(), trim=False)}")

		for epoch_step in range(args.epochs):
			while batched_data != None:
				if self.global_step.eval() % args.checkpoint_steps == 0 and self.global_step.eval() != 0:
					logger.info(
						"Epoch %d global step %d learning rate %.4f step-time %.2f perplexity: %.2f, kg_ppx: %.2f, kg_loss: %.4f, kg_acc: %.4f" % (
							epoch_step, self.global_step.eval(), self.learning_rate.eval(), time_step, np.exp(loss_step[0]),
							np.exp(loss_step[1]), loss_step[1], loss_step[2]))
					self.trainSummary(self.global_step.eval() // args.checkpoint_steps,
									  {'loss': loss_step[0], 'perplexity': np.exp(loss_step[0])})
					self.store_checkpoint(sess, '%s/checkpoint_latest/%s' % (args.model_dir, args.name), "latest", args.name)

					dev_loss = self.evaluate(sess, data, args.batch_size, "dev", lamb=args.lamb)
					self.devSummary(self.global_step.eval() // args.checkpoint_steps,
									{'loss': dev_loss[0], 'perplexity': np.exp(dev_loss[0])})

					if np.sum(loss_step) > max(previous_losses):
						sess.run(self.learning_rate_decay_op)
					if dev_loss[0] < best_valid:
						best_valid = dev_loss[0]
						self.store_checkpoint(sess, '%s/checkpoint_best/%s' % (args.model_dir, args.name), "best", args.name)

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

			batched_data['resp_length'] = [len(batched_data['resp'][0])]
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

			kg = []
			kg_h_length = []
			kg_hr_length = []
			kg_hrt_length = []
			kg_index = []

			for _ in range(len(resp)):
				post += batched_data['post']
				post_length += batched_data['post_length']
				prev_length += batched_data['prev_length']

				kg += batched_data['kg']
				kg_h_length += batched_data['kg_h_length']
				kg_hr_length += batched_data['kg_hr_length']
				kg_hrt_length += batched_data['kg_hrt_length']
				kg_index += batched_data['kg_index']

			batched_data['post'] = post
			batched_data['post_length'] = post_length
			batched_data['prev_length'] = prev_length

			batched_data['kg'] = kg
			batched_data['kg_h_length'] = kg_h_length
			batched_data['kg_hr_length'] = kg_hr_length
			batched_data['kg_hrt_length'] = kg_hrt_length
			batched_data['kg_index'] = kg_index

			_, _, loss, _, _ = self.inference(sess, batched_data, lamb=args.lamb)
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

		metric1 = data.get_teacher_forcing_metric()
		metric2 = data.get_inference_metric()
		data.restart("test", batch_size=args.batch_size, shuffle=False)
		batched_data = data.get_next_batch("test")

		for i in range(2):
			logger.info(f"post@{i}: {data.convert_ids_to_tokens(batched_data['post_allvocabs'][i].tolist(), trim=False)}")
			logger.info(f"resp@{i}: {data.convert_ids_to_tokens(batched_data['resp_allvocabs'][i].tolist(), trim=False)}")

		while batched_data != None:
			batched_responses_id, gen_log_prob, _, _, _ = self.inference(sess, batched_data, lamb=args.lamb)
			metric1_data = {'resp_allvocabs': np.array(batched_data['resp_allvocabs']),
							'resp_length': np.array(batched_data['resp_length']), 'gen_log_prob': np.array(gen_log_prob)}
			metric1.forward(metric1_data)
			batch_results = []
			for response_id in batched_responses_id:
				response_id_list = response_id.tolist()
				if data.eos_id in response_id_list:
					result_id = response_id_list[:response_id_list.index(data.eos_id) + 1]
				else:
					result_id = response_id_list
				batch_results.append(result_id)

			metric2_data = {'gen': np.array(batch_results), 'resp_allvocabs': np.array(batched_data['resp_allvocabs'])}
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

		logger.info("result output to %s." % test_file)
		return {key: val for key, val in res.items() if type(val) in [bytes, int, float]}
