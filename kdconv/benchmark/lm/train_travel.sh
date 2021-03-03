#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python run.py \
  --name=LM \
  --wv_path=/data/models/wordvector/chinese \
  --datapath=../../data/travel \
  --epoch=100 \
  --batch_size=32 \
  --max_sent_length=512 \
  --max_decoder_length=30 \
  --output_dir=/data/results/kdconv/output/travel/lm \
  --model_dir=/data/results/kdconv/model/travel/lm \
  --cache_dir=/data/results/kdconv/cache/travel/lm \
  --log_dir=/data/results/kdconv/log/travel/lm \
  --mode=train \
  --cache \
  --seed=42