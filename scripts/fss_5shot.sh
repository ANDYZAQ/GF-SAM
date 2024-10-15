#!/bin/bash

for fold in 0;
do
  CUDA_VISIBLE_DEVICES=8 \
  python main_eval.py \
    --benchmark fss \
    --nshot 5 \
    --fold ${fold} --log-root "output/fss_5shot/fold${fold}"
done
wait
