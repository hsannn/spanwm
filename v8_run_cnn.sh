#!/bin/bash

echo "Start SpamWM with llama3.1-8B! We're using cnn dataset!!"

python spanwm_embed_v8.py --dataset cnn --num_samples 200 --model meta-llama/Llama-3.1-8B

echo "End llama3.1-8B embedding for cnn dataset!"

echo "#####################################################################"

echo "Start SpamWM with llama3.2-3B! We're using cnn dataset!!"

python spanwm_embed_v8.py --dataset cnn --num_samples 200 --model meta-llama/Llama-3.2-3B

echo "End llama3.2-3B embedding for cnn dataset!"

echo "#####################################################################"

echo "Start SpamWM with qwen3-8b! We're using cnn dataset!!"

python spanwm_embed_v8.py --dataset cnn --num_samples 200 --model Qwen/Qwen3-8B

echo "End qwen3-8b embedding for cnn dataset!"

echo "#####################################################################"

echo "Start SpamWM with qwen3-4b! We're using cnn dataset!!"

python spanwm_embed_v8.py --dataset cnn --num_samples 200 --model Qwen/Qwen3-4B

echo "End qwen3-4b embedding for cnn dataset!"

