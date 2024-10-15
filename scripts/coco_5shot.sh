#!/bin/bash

for fold in 0 1 2 3;
do
  CUDA_VISIBLE_DEVICES=0 \
  python main_cus.py  \
    --benchmark coco \
    --nshot 5 \
    --fold ${fold} --log-root "output/coco_5shot/fold${fold}"
done
wait

