#!/bin/bash

# for fold in 0 1 2 3;
for fold in 1;
do
  CUDA_VISIBLE_DEVICES=9 \
  python main_eval.py \
    --benchmark pascal_part \
    --fold ${fold} --log-root "output/pascal_part/fold${fold}" \
    --visualize 1
done
wait