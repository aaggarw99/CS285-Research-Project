#!/bin/sh
# Baseline predictions without the use of any features
# python hw2/cs285/scripts/run_hw2.py \
#     --env_name ALE/Berzerk-v5 --ep_len 1500 --discount 0.95 \
#     -b 500 -s 128 -lr 5e-3 -rtg -rbs 4000 --nn_baseline \
#     --exp_name oom-test

# With features
python hw2/cs285/scripts/run_hw2.py \
    --env_name ALE/Berzerk-v5 --ep_len 1500 --discount 0.95 \
    -b 500 -s 128 -lr 5e-3 -rtg -rbs 3000 --nn_baseline \
    --exp_name features-test
