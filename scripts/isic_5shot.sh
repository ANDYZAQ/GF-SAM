#!/bin/bash

for fold in 0;
do
  CUDA_VISIBLE_DEVICES=8 \
  python main_eval.py \
    --benchmark isic \
    --nshot 5 \
     --fold ${fold} --log-root "output/isic_5shot/fold${fold}"
done
wait