#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python run.py \
  --name=LM \
  --wv_path=/data/models/wordvector/chinese \
  --datapath=../../data/music \
  --batch_size=32 \
  --max_sent_length=512 \
  --max_decoder_length=30 \
  --output_dir=/data/results/kdconv/output/music/lm \
  --model_dir=/data/results/kdconv/model/music/lm \
  --cache_dir=/data/results/kdconv/cache/music/lm \
  --log_dir=/data/results/kdconv/log/music/lm \
  --mode=test \
  --restore=best \
  --cache \
  --seed=42