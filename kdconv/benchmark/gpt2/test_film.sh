#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python run_GPTGen.py \
  --do_predict \
  --gpt_config_file=/data/models/ptms/CDial-GPT2_LCCC-base/config.json \
  --vocab_file=/data/models/ptms/CDial-GPT2_LCCC-base/vocab.txt \
  --init_checkpoint=/data/models/ptms/CDial-GPT2_LCCC-base/pytorch_model.bin \
  --name=GPT2Gen \
	--max_sent_length=256 \
	--num_turns=8 \
	--predict_batch_size=1 \
	--min_decoder_length=3 \
	--max_decoder_length=30 \
	--temperature=0.7 \
	--top_p=0.9 \
	--cache \
	--cache_dir=/data/results/kdconv/cache/film/gpt2gen \
	--datapath=../../data/film \
	--num_train_epochs=10.0 \
	--output_dir=/data/results/kdconv/output/film/gpt2gen \
  --model_dir=/data/results/kdconv/model/film/gpt2gen \
  --seed=42 \
  --is_relative      # 相对轮次编码