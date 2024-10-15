r""" iSAID-5i few-shot semantic segmentation dataset """
import os
from typing_extensions import override

import torch
import PIL.Image as Image
import numpy as np
import torchvision.transforms as transforms2
from .pascal import DatasetPASCAL


class DatasetISAID(DatasetPASCAL):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize, aug=False) -> None:
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 3
        self.nclass = 15
        self.benchmark = 'isaid'
        self.shot = shot
        self.use_original_imgsize = use_original_imgsize

        datapath = os.path.join(datapath, 'remote_sensing/iSAID_patches')

        if self.split == 'trn':
            self.img_path = os.path.join(datapath, 'train/images')
            self.ann_path = os.path.join(datapath, 'train/semantic_png')
        else:
            self.img_path = os.path.join(datapath, 'val/images')
            self.ann_path = os.path.join(datapath, 'val/semantic_png')

        self.aug = aug and (self.split == 'trn')
        if self.aug:
            self.tv2 = transforms2.Compose([
                transforms2.RandomHorizontalFlip(),
                transforms2.RandomRotation(30),
                # transforms2.RandomResizedCrop(size=256, scale=(0.5, 1.0))
            ])

        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    @override
    def __len__(self):
        return len(self.img_metadata)  # TODO: why hsnet use 100 for val

    @override
    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        # mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '_instance_color_RGB.png')))
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '_instance_color_RGB.png'))) 
        return mask

    @override
    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(os.path.join(self.img_path, img_name) + '.png')

    @override
    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join('matcher/data/splits/isaid/%s/fold%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata