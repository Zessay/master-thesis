#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python run.py \
  --name=hredmem \
  --wv_path=/data/models/wordvector/chinese \
  --datapath=../../data/film \
  --epoch=20 \
  --batch_size=32 \
  --max_sent_length=512 \
  --max_know_length=100 \
  --max_decoder_length=50 \
  --output_dir=/data/results/kdconv/output/film/hredmem \
  --model_dir=/data/results/kdconv/model/film/hredmem \
  --cache_dir=/data/results/kdconv/cache/film/hredmem \
  --log_dir=/data/results/kdconv/log/film/hredmem \
  --mode=train \
  --cache \
  --seed=42 \
  --lamb=1.0