#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python run_BERTRetrieval.py \
	--do_predict \
	--bert_config_file=/data/models/ptms/chinese_wwm_pytorch/config.json \
	--vocab_file=/data/models/ptms/chinese_wwm_pytorch/vocab.txt \
	--init_checkpoint=/data/models/ptms/chinese_wwm_pytorch \
	--name=BERTRetrieval \
	--num_choices=10 \
	--max_sent_length=256 \
	--num_turns=8 \
	--predict_batch_size=16 \
	--cache \
	--cache_dir=/data/results/kdconv/cache/music/bert_ret \
	--datapath=../../data/music \
	--num_train_epochs=5.0 \
	--output_dir=/data/results/kdconv/output/music/bert_ret \
  --model_dir=/data/results/kdconv/model/music/bert_ret