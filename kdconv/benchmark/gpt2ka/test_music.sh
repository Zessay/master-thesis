#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python run_GPTGen.py \
  --do_predict \
  --gpt_config_file=/data/models/ptms/CDial-GPT2_LCCC-base/config.json \
  --vocab_file=/data/models/ptms/CDial-GPT2_LCCC-base/vocab.txt \
  --init_checkpoint=/data/models/ptms/CDial-GPT2_LCCC-base/pytorch_model.bin \
  --wv_path=/data/models/wordvector/chinese \
  --name=GPT2GenKA \
	--max_sent_length=256 \
	--max_know_length=128 \
	--num_turns=8 \
	--predict_batch_size=1 \
	--min_decoder_length=3 \
	--max_decoder_length=30 \
	--temperature=0.7 \
	--top_p=0.9 \
	--cache \
	--cache_dir=/data/results/kdconv/cache/music/gpt2genka \
	--datapath=../../data/music \
	--num_train_epochs=10.0 \
	--output_dir=/data/results/kdconv/output/music/gpt2genka \
  --model_dir=/data/results/kdconv/model/music/gpt2genka \
  --seed=42 \
  --lamb=0.0 \
  --is_relative      # 相对轮次编码