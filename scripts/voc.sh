#!/bin/bash

for fold in 3;
# for fold in 0 1 2 3;
do
  CUDA_VISIBLE_DEVICES=5 \
  python main_eval.py  \
    --benchmark pascal \
    --fold ${fold} --log-root "output/pascal/fold${fold}" #\
    # --visualize 1
done
wait