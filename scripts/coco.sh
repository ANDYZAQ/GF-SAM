#!/bin/bash

# for fold in 0;
for fold in 0 1 2 3;
do
  CUDA_VISIBLE_DEVICES=0 \
  python main_eval.py  \
    --benchmark coco \
    --fold ${fold} --log-root "output/coco/fold${fold}"
done
wait
