# coding=utf-8
# @Author: 莫冉
# @Date: 2021-01-05
import os

import logging
import numpy as np
import tensorflow as tf
from myCoTK.dataloader import MyLM
from myCoTK.wordvector import TencentChinese
from utils import debug, try_cache

from .model import LM

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO)
logger = logging.getLogger(__name__)


def create_model(sess, data, args, embed):
    with tf.variable_scope(args.name):
        model = LM(data, args, embed)
        model.print_parameters()
        latest_dir = '%s/checkpoint_latest' % args.model_dir
        best_dir = '%s/checkpoint_best' % args.model_dir
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        if not os.path.isdir(latest_dir):
            os.mkdir(latest_dir)
        if not os.path.isdir(best_dir):
            os.mkdir(best_dir)
        if tf.train.get_checkpoint_state(
                latest_dir, args.name) and args.restore == "last":
            logger.info(
                "Reading model parameters from %s" %
                tf.train.latest_checkpoint(
                    latest_dir, args.name))
            model.latest_saver.restore(
                sess, tf.train.latest_checkpoint(
                    latest_dir, args.name))
        else:
            if tf.train.get_checkpoint_state(
                    best_dir, args.name) and args.restore == "best":
                logger.info(
                    'Reading model parameters from %s' %
                    tf.train.latest_checkpoint(
                        best_dir, args.name))
                model.best_saver.restore(
                    sess, tf.train.latest_checkpoint(
                        best_dir, args.name))
            else:
                logger.info("Created model with fresh parameters.")
                global_variable = [
                    gv for gv in tf.global_variables() if args.name in gv.name]
                sess.run(tf.variables_initializer(global_variable))

    return model


def main(args):
    if args.debug:
        debug()

    if args.cuda:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
    else:
        config = tf.ConfigProto(device_count={'GPU': 0})
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    data_class = MyLM
    wordvec_class = TencentChinese
    logger.info("模型侧加载数据")
    if args.cache:
        if not os.path.isdir(args.cache_dir):
            os.mkdir(args.cache_dir)
        data = try_cache(data_class,
                         {"file_id": args.datapath,
                          "max_sent_length": args.max_sent_length},
                         args.cache_dir)
        vocab = data.vocab_list
        logger.info("加载词向量")
        embed = try_cache(
            lambda wv, ez, vl: wordvec_class(wv).load_matrix(ez, vl),
            (args.wv_path,
             args.embedding_size,
             vocab),
            args.cache_dir,
            wordvec_class.__name__)
    else:
        data = data_class(file_id=args.datapath,
                          max_sent_length=args.max_sent_length)
        logger.info("定义并加载词向量文件")
        wv = wordvec_class(args.wv_path)
        vocab = data.vocab_list
        embed = wv.load_matrix(args.embedding_size, vocab)

    embed = np.array(embed, dtype=np.float32)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    with tf.Session(config=config) as sess:
        model = create_model(sess, data, args, embed)
        if args.mode == "train":
            logger.info("开始训练...")
            model.train_process(sess, data, args)
        else:
            logger.info("开始测试...")
            model.test_process(sess, data, args)
