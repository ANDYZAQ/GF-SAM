#!/bin/bash

for fold in 2;
do
  CUDA_VISIBLE_DEVICES=2 \
  python main_eval.py \
    --benchmark isaid \
    --fold ${fold} --log-root "output/isaid/fold${fold}" \
    --visualize 1
done
wait