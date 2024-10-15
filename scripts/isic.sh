#!/bin/bash

for fold in 0;
do
  CUDA_VISIBLE_DEVICES=7 \
  python main_eval.py \
    --benchmark isic \
    --fold ${fold} --log-root "output/isic/fold${fold}"
done
wait