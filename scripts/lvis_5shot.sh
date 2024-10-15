#!/bin/bash

for fold in 0 1 2 3 4 5 6 7 8 9;
do
  CUDA_VISIBLE_DEVICES=2 \
  python main_eval.py  \
    --benchmark lvis \
    --nshot 5 \
     --fold ${fold} --log-root "output/lvis_5shot/fold${fold}"
done
wait