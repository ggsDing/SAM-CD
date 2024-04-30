# SAM-CD
Pytorch codes of **Adapting Segment Anything Model for Change Detection in HR Remote Sensing Images** [[paper](https://ieeexplore.ieee.org/document/10443350)]

![alt text](https://github.com/ggsDing/SAM-CD/blob/main/flowchart.png)

The SAM-CD adopts [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) as the visual encoder with some modifications.

## 2024-4-30 Update:

SAM-CD now supports access to [efficientSAM](https://github.com/yformer/EfficientSAM). Check the updated model at ```models/effSAM_CD.py``` (prior installation of efficientSAM at the project folder is required). However, direct integration of efficientSAM may cause an accuracy drop, so there is space to further improve the SAM-CD architecture.

## How to Use
1. Installation
   * Install [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) following the instructions.
   * Modify the Ultralytics source files following the instructions at: ['SAM-CD/models/FastSAM/README.md'](https://github.com/ggsDing/SAM-CD/blob/main/models/FastSAM/README.md). 

2. Dataset preparation.
   * Please split the data into training, validation and test sets and organize them as follows:
```
      YOUR_DATA_DIR
      ├── ...
      ├── train
      │   ├── A
      │   ├── B
      │   ├── label
      ├── val
      │   ├── A
      │   ├── B
      │   ├── label
      ├── test
      │   ├── A
      │   ├── B
      │   ├── label
```

   * Find change line 13 in [SAM-CD/datasets/Levir_CD.py](https://github.com/ggsDing/SAM-CD/blob/main/datasets/Levir_CD.py) (or other data-loading .py files), change `/YOUR_DATA_ROOT/` to your local dataset directory.

3. Training
   
   classic CD training:
   `python train_CD.py`
   
   training CD with the proposed task-agnostic semantic learning:
   `python train_SAM_CD.py`
   
   line 16-45 are the major training args, which can be changed to load different datasets, models and adjust the training settings.

5. Inference and evaluation
   
   inference on test sets: set the chkpt_path and run
   
   `python pred_CD.py`
   
   evaluation of accuracy: set the prediction dir and GT dir, and run
   
   `python eval_CD.py`
   
(More details to be added...)


## Dataset Download

In the following, we summarize links to some frequently used CD datasets:

* [LEVIR-CD](https://justchenhao.github.io/LEVIR/)
* [WHU-CD](https://study.rsgis.whu.edu.cn/pages/download/) [(baidu)](https://pan.baidu.com/s/1A0_xbV4ZktWCbL3j94CInA?pwd=WHCD )
* [CLCD (Baidu)](https://pan.baidu.com/s/1iZtAq-2_vdqoz1RnRtivng?pwd=CLCD)
* [S2Looking](https://github.com/S2Looking/Dataset)
* [SYSU-CD](https://github.com/liumency/SYSU-CD)

## Pretrained Models

For readers to easily evaluate the accuracy, we provide the trained weights of the SAM-CD.

[Drive](https://drive.google.com/drive/folders/14tNtID43o-LHs8VaMK5jai1Uf8NqMDAW?usp=sharing)  
[Baidu](https://pan.baidu.com/s/1V25TFGL5V05ZB5ttFXFSEA?pwd=SMCD) (pswd: SMCD)


## Cite SAM-CD

If you find this work useful or interesting, please consider citing the following BibTeX entry.

```
@article{ding2024adapting,
title={Adapting Segment Anything Model for Change Detection in HR Remote Sensing Images},
author={Ding, Lei and Zhu, Kun and Peng, Daifeng and Tang, Hao and Yang, Kuiwu and Bruzzone, Lorenzo},
journal={IEEE Transactions on Geoscience and Remote Sensing}, 
year={2024},
volume={62},
pages={1-11},
doi={10.1109/TGRS.2024.3368168}
}

```
