#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python run_GPTGen.py \
  --do_train \
  --gpt_config_file=/data/models/ptms/CDial-GPT2_LCCC-base/config.json \
  --vocab_file=/data/models/ptms/CDial-GPT2_LCCC-base/vocab.txt \
  --init_checkpoint=/data/models/ptms/CDial-GPT2_LCCC-base/pytorch_model.bin \
  --name=GPT2Gen \
  --num_choices=10 \
	--max_sent_length=256 \
	--num_turns=8 \
	--train_batch_size=16 \
	--learning_rate=2e-5 \
	--cache \
	--cache_dir=/data/results/kdconv/cache/travel/gpt2gen \
	--datapath=../../data/travel \
	--num_train_epochs=5.0 \
	--warmup_proportion=0.1 \
	--output_dir=/data/results/kdconv/output/travel/gpt2gen \
  --model_dir=/data/results/kdconv/model/travel/gpt2gen \
  --gradient_accumulation_steps=8 \
  --seed=42 \
  --is_relative      # 相对轮次编码