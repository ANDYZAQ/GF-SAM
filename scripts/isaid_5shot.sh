#!/bin/bash

for fold in 1;
do
  CUDA_VISIBLE_DEVICES=1 \
  python main_eval.py \
    --benchmark isaid \
    --nshot 5 \
     --fold ${fold} --log-root "output/isaid_5shot/fold${fold}"
done
wait