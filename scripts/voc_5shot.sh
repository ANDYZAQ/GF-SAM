#!/bin/bash

# for fold in 0;
for fold in 0 1 2 3;
do
  CUDA_VISIBLE_DEVICES=2 \
  python main_eval.py  \
    --benchmark pascal \
    --nshot 5 \
    --fold ${fold} --log-root "output/pascal_5shot/fold${fold}"
done
wait