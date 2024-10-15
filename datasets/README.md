## Preparing Few-Shot Segmentation Datasets
Download following datasets:


> #### 1. COCO-20<sup>i</sup>
> Download COCO2014 train/val images and annotations: 
> ```bash
> wget http://images.cocodataset.org/zips/train2014.zip
> wget http://images.cocodataset.org/zips/val2014.zip
> wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
> ```

> #### 2. Pascal-5<sup>i</sup>
> Download PASCAL VOC 2012 devkit (train/val data):: 
> ```bash
> wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
> ```
> Download PASCAL VOC2012 SDS extended mask annotations from [Google Drive](https://drive.google.com/file/d/10zxG2VExoEZUeyQl_uXga2OWHjGeZaf2/view?usp=sharing)

> #### 3. FSS-1000
> Download FSS-1000 images and annotations from this [Google Drive](https://drive.google.com/file/d/1Fn-cUESMMF1pQy8Xff-vPQvXJdZoUlP3/view?usp=sharing).

> #### 4. LVIS-92<sup>i</sup>
> Download COCO2017 train/val images: 
> ```bash
> wget http://images.cocodataset.org/zips/train2017.zip
> wget http://images.cocodataset.org/zips/val2017.zip
> ```
> Download LVIS-92<sup>i</sup> extended mask annotations from Google Drive: [lvis.zip](https://drive.google.com/file/d/1itJC119ikrZyjHB9yienUPD0iqV12_9y/view?usp=sharing).


> #### 5. PACO-Part
> Download COCO2017 train/val images: 
> ```bash
> wget http://images.cocodataset.org/zips/train2017.zip
> wget http://images.cocodataset.org/zips/val2017.zip
> ```
> Download PACO-Part extended mask annotations from Google Drive: [paco.zip](https://drive.google.com/file/d/1VEXgHlYmPVMTVYd8RkT6-l8GGq0G9vHX/view?usp=sharing).

> #### 6. Pascal-Part
> Download VOC2010 train/val images: 
> ```bash
> wget http://roozbehm.info/pascal-parts/trainval.tar.gz
> wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
> ```
> Download Pascal-Part extended mask annotations from Google Drive: [pascal.zip](https://drive.google.com/file/d/1WaM0VM6I9b3u3v3w-QzFLJI8d3NRumTK/view?usp=sharing).

> #### 7. Deepglobe
> Download Deepglobe from [Deepglobe](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset), or download preprocess data from [Google Drive](https://drive.google.com/file/d/10qsi1NRyFKFyoIq1gAKDab6xkbE0Vc74/view?usp=sharing). 

> #### 8. ISIC2018
> Download ISIC from [ISIC Challenge 2018](https://challenge.isic-archive.com/data#2018). 

> #### 9. iSAID-5<sup>i</sup>
> Download iSAID-5<sup>i</sup> from [Google Drive](https://drive.google.com/drive/folders/1-URr9fX0v6_-Yo3B7St8UFNHiPWpXxnC?usp=sharing). 


Create a directory 'datasets' for the above datasets and appropriately place each dataset to have following directory structure:

    datasets/
    ├── COCO2014/           
    │   ├── annotations/
    │   │   ├── train2014/
    │   │   └── val2014/
    │   ├── train2014/
    │   ├── val2014/
    │   └── splits
    │   │   ├── trn/
    │   │   └── val/
    ├── VOC2012/
    │   ├── Annotations/
    │   ├── ImageSets/
    │   ├── ...
    │   └── SegmentationClassAug/
    ├── FSS-1000/
    │   ├── data/
    │   │   ├── ab_wheel/
    │   │   ├── ...
    │   │   └── zucchini/
    │   └── splits/   
    │   │   ├── test.text
    │   │   ├── trn.txt
    │   │   └── val.txt
    ├── LVIS/
    │   ├── coco/
    │   │   ├── train2017/
    │   │   └── val2017/
    │   ├── lvis_train.pkl
    │   └── lvis_val.pkl
    ├── PACO-Part/
    │   ├── coco/
    │   │   ├── train2017/
    │   │   └── val2017/
    │   └── paco/
    │       ├── paco_part_train.pkl
    │       └── paco_part_val.pkl
    ├── Pascal-Part/  
    │   └── VOCdevkit/
    │       └── VOC2010/
    │           ├── Annotations_Part_json_merged_part_classes/
    │           ├── JPEGImages/
    │           └── all_obj_part_to_image.json
    ├── Deepglobe/  
    │   ├── 1/
    │   │   ├── test/
    │   │   │   ├── groundtruth/
    │   │   │   └── origin/
    │   ├── 2/
    │   ├── ...
    │   └── 6/
    ├── ISIC/  
    │   ├── ISIC2018_Task1_Training_GroundTruth/
    │   └── ISIC2018_Task1-2_Training_Input/
    └── remote_sensing/  
        ├── iSAID_patches/
        │   ├── train/
        │   │   ├── images/
        │   │   ├── ...
        │   │   └── train_list/
        │   ├── val/
        │   │   ├── images/
        │   │   ├── ...
        │   │   └── val_list/
        └── label.xslx
        
