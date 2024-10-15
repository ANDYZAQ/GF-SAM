#!/bin/bash

for fold in 0;
do
  CUDA_VISIBLE_DEVICES=3 \
  python main_eval.py \
    --benchmark fss \
    --fold ${fold} --log-root "output/fss/fold${fold}" \
    --visualize 1
done
wait