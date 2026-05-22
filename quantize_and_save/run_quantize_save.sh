#!/bin/bash

SAVE_DIR="model_zoo/qsa2/"

CUDA_VISIBLE_DEVICES=0 python quantize_save.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --bits 2 \
    --iter 5 \
    --rank 64 \
    --save_dir $SAVE_DIR

