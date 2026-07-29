#!/bin/bash

echo "Start SpamWM with qwen3-8b! We're using wmt16 dataset!!"

python spanwm_embed_v8.py --dataset wmt16 --num_samples 200 --model Qwen/Qwen3-8B

echo "End qwen3-8b embedding for wmt16 dataset!"

echo "#####################################################################"

echo "Start SpamWM with qwen3-4b! We're using wmt16 dataset!!"

python spanwm_embed_v8.py --dataset wmt16 --num_samples 200 --model Qwen/Qwen3-4B

echo "End qwen3-4b embedding for wmt16 dataset!"

