#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES='0'

python run.py \
  --name=seq2seqmem \
  --wv_path=/data/models/wordvector/chinese \
  --datapath=../../data/travel \
  --batch_size=32 \
  --max_sent_length=512 \
  --max_know_length=100 \
  --max_decoder_length=50 \
  --output_dir=/data/results/kdconv/output/travel/seq2seqmem \
  --model_dir=/data/results/kdconv/model/travel/seq2seqmem \
  --cache_dir=/data/results/kdconv/cache/travel/seq2seqmem \
  --log_dir=/data/results/kdconv/log/travel/seq2seqmem \
  --mode=test \
  --restore=best \
  --cache \
  --seed=42 \
  --lamb=1.0