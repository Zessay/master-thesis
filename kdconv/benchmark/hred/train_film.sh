#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python run.py \
  --name=hred \
  --wv_path=/data/models/wordvector/chinese \
  --datapath=../../data/film \
  --epoch=50 \
  --batch_size=32 \
  --max_sent_length=256 \
  --max_decoder_length=30 \
  --output_dir=/data/results/kdconv/output/film/hred \
  --model_dir=/data/results/kdconv/model/film/hred \
  --cache_dir=/data/results/kdconv/cache/film/hred \
  --log_dir=/data/results/kdconv/log/film/hred \
  --mode=train \
  --cache \
  --seed=42