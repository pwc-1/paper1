#!/usr/bin/env sh
cd MTL-AQA
mkdir -p logs
mkdir -p ckpts
CUDA_VISIBLE_DEVICES=$1 python3 -u main.py ${@:2}
