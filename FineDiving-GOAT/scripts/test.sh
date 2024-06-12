#!/usr/bin/env sh
mkdir -p logs
CUDA_VISIBLE_DEVICES=$1 python3 -u main.py --test ${@:2}