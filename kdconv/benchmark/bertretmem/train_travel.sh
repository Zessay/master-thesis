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
	--learning_rate=5e-5 \
	--lamb=0.0 \
	--cache \
	--cache_dir=/data/results/kdconv/cache/travel/bert_ret_mem \
	--datapath=../../data/travel \
	--num_train_epochs=8.0 \
	--warmup_proportion=0.1 \
	--output_dir=/data/results/kdconv/output/travel/bert_ret_mem \
  --model_dir=/data/results/kdconv/model/travel/bert_ret_mem \
  --gradient_accumulation_steps=8 \
  --seed=42