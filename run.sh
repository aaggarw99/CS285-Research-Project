#!/bin/sh
python hw2/cs285/scripts/run_hw2.py \
    --env_name ALE/Berzerk-v5 --ep_len 1000 --discount 0.95 \
    -b 1000 -s 128 -lr 5e-3 -rtg --nn_baseline \
    --exp_name oom-test
