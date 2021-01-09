# coding=utf-8
# @Author: 莫冉
# @Date: 2021-01-08
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import warnings
warnings.filterwarnings("ignore")

import argparse
import time

from utils import Storage


def run(*argv):
	parser = argparse.ArgumentParser(description='A hred model')
	args = Storage()

	parser.add_argument('--name', type=str, default='hred',
		help='The name of your model, used for variable scope and tensorboard, etc. Default: runXXXXXX_XXXXXX (initialized by current time)')
	parser.add_argument('--restore', type=str, default='best',
		help='Checkpoints name to load. "last" for last checkpoints, "best" for best checkpoints on dev. Attention: "last" and "best" wiil cause unexpected behaviour when run 2 models in the same dir at the same time. Default: None (don\'t load anything)')
	parser.add_argument('--mode', type=str, default="train",
		help='"train" or "test". Default: train')
	parser.add_argument('--dataset', type=str, default='MyHRED',
		help='Dataloader class. Default: UbuntuCorpus')
	parser.add_argument('--datapath', type=str, default='../data/film',
		help='Directory for data set. Default: UbuntuCorpus')
	parser.add_argument('--epoch', type=int, default=20,
		help="Epoch for trainning. Default: 100")
	parser.add_argument('--batch_size', type=int, default=32,
		help="The batch size of data when train or test.")
	parser.add_argument('--max_sent_length', type=int, default=512,
		help="The max encoded sent length when train.")
	parser.add_argument('--max_decoder_length', type=int, default=50,
		help="The max decoded sent length when inference.")
	parser.add_argument('--num_turns', type=int, default=8,
		help="The max number of turns of the post field.")
	parser.add_argument('--wv_class', type=str, default='TencentChinese',
		help="Wordvector class, none for not using pretrained wordvec. Default: Glove")
	parser.add_argument('--wv_path', type=str, default='wordvector/chinese',
		help="Directory for pretrained wordvector. Default: resources://Glove300d")

	parser.add_argument('--output_dir', type=str, default="./output/film",
		help='Output directory for test output. Default: ./output')
	parser.add_argument('--log_dir', type=str, default="./tensorboard/film",
		help='Log directory for tensorboard. Default: ./tensorboard')
	parser.add_argument('--model_dir', type=str, default="./model/film",
		help='Checkpoints directory for model. Default: ./model')
	parser.add_argument('--cache_dir', type=str, default="./cache/film",
		help='Checkpoints directory for cache. Default: ./cache')
	parser.add_argument('--cpu', action="store_true",
		help='Use cpu.')
	parser.add_argument('--debug', action='store_true',
		help='Enter debug mode (using ptvsd).')
	parser.add_argument('--cache', action='store_true',
		help='Use cache for speeding up load data and wordvec. (It may cause problems when you switch dataset.)')
	parser.add_argument('--seed', type=int, default=42,
		help="The random seed in the train process.")
	cargs = parser.parse_args(argv)

	# Editing following arguments to bypass command line.
	args.name = cargs.name or time.strftime("run%Y%m%d_%H%M%S", time.localtime())
	args.restore = cargs.restore
	args.mode = cargs.mode
	args.dataset = cargs.dataset
	args.datapath = cargs.datapath
	args.epochs = cargs.epoch
	args.batch_size = cargs.batch_size
	args.wv_class = cargs.wv_class
	args.wv_path = cargs.wv_path
	args.output_dir = cargs.output_dir
	args.log_dir = cargs.log_dir
	args.model_dir = cargs.model_dir
	args.cache_dir = cargs.cache_dir
	args.debug = cargs.debug
	args.cache = cargs.cache
	args.cuda = not cargs.cpu
	args.seed = cargs.seed
	args.max_sent_length = cargs.max_sent_length
	args.max_decoder_length = cargs.max_decoder_length
	args.num_turns = cargs.num_turns

	args.softmax_samples = 512
	args.embedding_size = 200
	args.eh_size = 200
	args.ch_size = 200
	args.dh_size = 200
	args.lr = 1e-3
	args.lr_decay = 0.99
	args.grad_clip = 5.0
	args.show_sample = [0]
	args.checkpoint_steps = 100
	args.checkpoint_max_to_keep = 5

	import random
	random.seed(args.seed)

	from .main import main

	main(args)

if __name__ == '__main__':
	import sys
	run(*sys.argv[1:])
