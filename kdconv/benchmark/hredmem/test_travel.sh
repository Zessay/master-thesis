#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python run.py \
  --name=hredmem \
  --wv_path=/data/models/wordvector/chinese \
  --datapath=../../data/travel \
  --batch_size=32 \
  --max_sent_length=512 \
  --max_know_length=100 \
  --max_decoder_length=30 \
  --output_dir=/data/results/kdconv/output/travel/hredmem \
  --model_dir=/data/results/kdconv/model/travel/hredmem \
  --cache_dir=/data/results/kdconv/cache/travel/hredmem \
  --log_dir=/data/results/kdconv/log/travel/hredmem \
  --mode=test \
  --restore=best \
  --cache \
  --seed=42 \
  --lamb=1.0