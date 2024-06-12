#!/usr/bin/env sh
cd MTL-AQA
mkdir -p logs
CUDA_VISIBLE_DEVICES=$1 python3 -u test.py ${@:2}