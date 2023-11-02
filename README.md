# SAM-CD
Pytorch codes of 'Adapting Segment Anything Model for Change Detection in HR Remote Sensing Images' [[paper](http://arxiv.org/abs/2309.01429)]

![alt text](https://github.com/ggsDing/SAM-CD/blob/main/flowchart.png)

Our SAM-CD adopts [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) as the visual encoder with some modifications.


**How to Use**
1. Installation
   1) Install [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) following the instructions.
   2) Find 'SAM-CD/models/FastSAM/changes.rec'. Open it in plain text, and modify the Ultralytics source files following the instructions.

2. Dataset preparation.
   Please split the data into training, validation and test sets and organize them as follows:
   
>YOUR_DATA_DIR
>  - train
>    - A
>    - B
>  - val
>    - A
>    - B
>  - test
>    - A
>    - B
>  - label

    Then, find change line 13 in SAM-CD/datasets/Levir_CD.py (or other dataloading .py files), change '/YOUR_DATA_ROOT/' to your own dataset directory.




(A guide to use is to be added...)

