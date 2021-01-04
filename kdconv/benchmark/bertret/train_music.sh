#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'


python run_BERTRetrieval.py \
	--do_train \
	--bert_config_file=/data/models/ptms/chinese_wwm_pytorch/config.json \
	--vocab_file=/data/models/ptms/chinese_wwm_pytorch/vocab.txt \
	--init_checkpoint=/data/models/ptms/chinese_wwm_pytorch \
	--name=BERTRetrieval \
	--num_choices=10 \
	--train_batch_size=16 \
	--learning_rate=5e-5 \
	--cache \
	--cache_dir=/data/results/kdconv/cache/music/bert_ret \
	--datapath=../../data/music \
	--num_train_epochs=5.0 \
	--warmup_proportion=0.1 \
	--output_dir=/data/results/kdconv/output/music/bert_ret \
  --model_dir=/data/results/kdconv/model/music/bert_ret \
  --gradient_accumulation_steps=8 \
  --seed=42