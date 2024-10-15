#!/bin/bash


for fold in 0;
# for fold in 0 1 2 3 4 5 6 7 8 9;
do
  CUDA_VISIBLE_DEVICES=9 \
  python main_eval.py \
    --benchmark lvis \
    --fold ${fold} --log-root "output/lvis/fold${fold}" \
    --visualize 1
done
wait