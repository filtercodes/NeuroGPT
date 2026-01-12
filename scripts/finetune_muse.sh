#!/bin/bash
# Fine-tune NeuroGPT on Muse Data (Domain Adaptation / CSM)
# Uses the pre-trained foundation model weights (6 layers)

python3 ../src/train_gpt.py \
    --run-name='muse_finetune_CSM' \
    --training-style='CSM_causal' \
    --train-data-path='../inputs/muse_tensors/' \
    --data-list='../inputs/muse_list.csv' \
    --pretrained-model='../pretrained_model/pytorch_model.bin' \
    --num-hidden-layers=6 \
    --num-encoder-layers=6 \
    --embedding-dim=1024 \
    --chunk_len=500 \
    --num_chunks=8 \
    --chunk_ovlp=50 \
    --n-filters-time=40 \
    --filter-time-length=25 \
    --pool-time-length=75 \
    --stride-avg-pool=15 \
    --training-steps=5000 \
    --eval_every_n_steps=500 \
    --log-every-n-steps=100 \
    --per-device-training-batch-size=8 \
    --per-device-validation-batch-size=8 \
    --learning-rate=5e-5 \
    --optim="adamw_torch" \
    --fp16=False \
    --do-normalization=True \
    --num-workers=0 \
    --use-encoder=True \
    --freeze-encoder=False \
    --ft-only-encoder=False