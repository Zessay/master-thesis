#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python run.py \
  --name=hred \
  --wv_path=/data/models/wordvector/chinese \
  --datapath=../../data/travel \
  --batch_size=32 \
  --max_sent_length=512 \
  --max_decoder_length=50 \
  --output_dir=/data/results/kdconv/output/travel/hred \
  --model_dir=/data/results/kdconv/model/travel/hred \
  --cache_dir=/data/results/kdconv/cache/travel/hred \
  --log_dir=/data/results/kdconv/log/travel/hred \
  --mode=test \
  --restore=best \
  --cache \
  --seed=42