#!/bin/bash

for fold in 0;
do
  CUDA_VISIBLE_DEVICES=9 \
  python main_eval.py \
    --benchmark deepglobe \
    --fold ${fold} --log-root "output/deepglobe/fold${fold}" \
    --visualize 1
done
wait