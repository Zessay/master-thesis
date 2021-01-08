#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python run.py \
  --name=seq2seq \
  --wv_path=/data/models/wordvector/chinese \
  --datapath=../../data/travel \
  --epoch=20 \
  --batch_size=32 \
  --max_sent_length=512 \
  --max_decoder_length=50 \
  --output_dir=/data/results/kdconv/output/travel/seq2seq \
  --model_dir=/data/results/kdconv/model/travel/seq2seq \
  --cache_dir=/data/results/kdconv/cache/travel/seq2seq \
  --log_dir=/data/results/kdconv/log/travel/seq2seq \
  --mode=train \
  --cache \
  --seed=42