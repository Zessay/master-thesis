#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python run.py \
  --name=hred \
  --wv_path=/data/models/wordvector/chinese \
  --datapath=../../data/film \
  --batch_size=32 \
  --max_sent_length=512 \
  --max_decoder_length=50 \
  --output_dir=/data/results/kdconv/output/film/hred \
  --model_dir=/data/results/kdconv/model/film/hred \
  --cache_dir=/data/results/kdconv/cache/film/hred \
  --log_dir=/data/results/kdconv/log/film/hred \
  --mode=test \
  --restore=best \
  --cache \
  --seed=42