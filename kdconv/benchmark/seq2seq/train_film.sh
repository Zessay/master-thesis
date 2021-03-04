#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python run.py \
  --name=seq2seq \
  --wv_path=/data/models/wordvector/chinese \
  --datapath=../../data/film \
  --epoch=50 \
  --batch_size=32 \
  --max_sent_length=256 \
  --max_decoder_length=30 \
  --output_dir=/data/results/kdconv/output/film/seq2seq \
  --model_dir=/data/results/kdconv/model/film/seq2seq \
  --cache_dir=/data/results/kdconv/cache/film/seq2seq \
  --log_dir=/data/results/kdconv/log/film/seq2seq \
  --mode=train \
  --cache \
  --seed=42