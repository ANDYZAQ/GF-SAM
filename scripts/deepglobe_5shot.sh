#!/bin/bash

for fold in 0;
do
  CUDA_VISIBLE_DEVICES=2 \
  python main_eval.py \
    --benchmark deepglobe \
    --nshot 5 \
     --fold ${fold} --log-root "output/deepglobe_5shot/fold${fold}"
done
wait