python train.py \
  --pretrained \
  --model_checkpoint=../../../models/LCCC_GPT_base \
  --data_path=../../../corpus/toy/toy_data.json \
  --dataset_cache=../../../corpus/toy/ \
  --scheduler=linear