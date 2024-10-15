#!/bin/bash

for fold in 0;
# for fold in 0 1 2 3;
do
  CUDA_VISIBLE_DEVICES=9 \
  python  main_eval.py \
    --benchmark paco_part \
    --fold ${fold}  --log-root "output/paco/fold${fold}" \
    --visualize 1
done
wait
