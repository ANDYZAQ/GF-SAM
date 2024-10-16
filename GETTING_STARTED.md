## Getting Started with GFSAM


### Prepare models

Download the model weights of [DINOv2](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) and [SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), and organize them as follows.
```
models/
    dinov2_vitl14_pretrain.pth
    sam_vit_h_4b8939.pth
```


### Test One-shot Semantic Segmentation

You can test one-shot semantic segmentation performance of GF-SAM on COCO-20<sup>i</sup>, run:

```
python main_eval.py  \
    --benchmark coco \
    --nshot 1 \
    --fold 0 --log-root "output/coco/fold0"
```

* You can replace `--benchmark coco` with `--benchmark lvis` to test LVIS-92<sup>i</sup>.
* You can replace `--nshot 1` with `--nshot 5` to test 5-shot performance on COCO-20<sup>i</sup>.
* You can find more commands in `scripts/` for other datasets.


