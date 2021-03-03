#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python run.py \
  --name=seq2seq \
  --wv_path=/data/models/wordvector/chinese \
  --datapath=../../data/music \
  --epoch=100 \
  --batch_size=32 \
  --max_sent_length=512 \
  --max_decoder_length=30 \
  --output_dir=/data/results/kdconv/output/music/seq2seq \
  --model_dir=/data/results/kdconv/model/music/seq2seq \
  --cache_dir=/data/results/kdconv/cache/music/seq2seq \
  --log_dir=/data/results/kdconv/log/music/seq2seq \
  --mode=train \
  --cache \
  --seed=42