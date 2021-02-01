#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'


python run_BERTRetrieval.py \
	--do_train \
	--bert_config_file=/data/models/ptms/chinese_wwm_pytorch/config.json \
	--vocab_file=/data/models/ptms/chinese_wwm_pytorch/vocab.txt \
	--init_checkpoint=/data/models/ptms/chinese_wwm_pytorch/pytorch_model.bin \
	--wv_path=/data/models/wordvector/chinese \
	--name=BERTMemRetrieval \
	--num_choices=10 \
	--max_sent_length=256 \
	--max_know_length=128 \
	--num_turns=8 \
	--train_batch_size=16 \
	--learning_rate=1e-4 \
	--lamb=0.2 \
	--cache \
	--cache_dir=/data/results/kdconv/cache/music/bert_ret_mem \
	--datapath=../../data/music \
	--num_train_epochs=10.0 \
	--warmup_proportion=0.1 \
	--output_dir=/data/results/kdconv/output/music/bert_ret_mem \
  --model_dir=/data/results/kdconv/model/music/bert_ret_mem \
  --gradient_accumulation_steps=8 \
  --seed=42