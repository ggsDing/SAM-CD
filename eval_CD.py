import os
import math
import numpy as np
from skimage import io, measure
from scipy import stats
from metric_tool import get_mIoU

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.TP = None
        self.TN = None
        self.FP = None
        self.FN = None
        self.count = None

    def initialize(self, TP, TN, FP, FN):    
        self.TP = float(TP)
        self.TN = float(TN)
        self.FP = float(FP)
        self.FN = float(FN)
        self.count = 1
        self.initialized = True

    def update(self, TP, TN, FP, FN):
        if not self.initialized:
            self.initialize(TP, TN, FP, FN)
        else:
            self.add(TP, TN, FP, FN)

    def add(self, TP, TN, FP, FN):
        self.TP += float(TP)
        self.TN += float(TN)
        self.FP += float(FP)
        self.FN += float(FN)
        self.count += 1        
        
    def val(self):
        return self.TP, self.TN, self.FP, self.FN

def align_dims(np_input, expected_dims=2):
    dim_input = len(np_input.shape)
    np_output = np_input
    if dim_input>expected_dims:
        np_output = np_input.squeeze(0)
    elif dim_input<expected_dims:
        np_output = np.expand_dims(np_input, 0)       
    assert len(np_output.shape) == expected_dims
    return np_output

def index2int(pred):
    pred = pred*255
    pred = np.asarray(pred, dtype='uint8')
    return pred

def calc_TP(pred, label):
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)
    pred = (pred>= 0.5)
    label = (label>= 0.5)
    
    GT = (label).sum()
    TP = (pred * label).sum()
    FP = (pred * (1-label)).sum()
    FN = ((1-pred) * (label)).sum()
    TN = ((1-pred) * (1-label)).sum()
    return TP, TN, FP, FN

if __name__ == '__main__':
    GT_dir = '/.../levir_CD/label/'
    pred_dir = '/.../SAM_CD/eval/Levir_CD/SAM_CD/'
      
    info_txt_path = os.path.join(pred_dir, 'info.txt')
    f = open(info_txt_path, 'w+')
    acc_meter = AverageMeter()
    
    data_list = os.listdir(pred_dir)
    num_valid = 0
    for idx, it in enumerate(data_list):
        if it[-4:]=='.png' or it[-4:]=='.jpg': num_valid+=1
    
    preds = []    
    GTs = []
    for idx, it in enumerate(data_list):
        if it[-4:]=='.png' or it[-4:]=='.jpg':
            pred_path = os.path.join(pred_dir, it)
            pred = io.imread(pred_path)
            h,w = pred.shape[:2]
            #pred = pred[:,:,0]/255.0
            GT_path = os.path.join(GT_dir, it[:-4]+'.png')
            GT = io.imread(GT_path)
            pred = pred//255 #pred.clip(max=1)
            GT = GT//255 #.clip(max=1)
            preds.append(pred)
            GTs.append(GT)
            
            #acc, precision, recall, F1, IoU = binary_accuracy(pred, GT)
            TP, TN, FP, FN = calc_TP(pred, GT[:h,:w])
            acc_meter.update(TP, TN, FP, FN)
            if not idx%10: print('Eval idx %d/%d processed.'%(idx, num_valid))
    
    TP, TN, FP, FN = acc_meter.val()
    precision = TP / (TP+FP+1e-10)
    recall = TP / (TP+FN+1e-10)
    IoU0 = TP / (FP+TP+FN+1e-10)
    IoU1 = TN / (FP+TN+FN+1e-10)
    mIoU = (IoU0+IoU1)/2
    acc = (TP+TN) / (TP+FP+FN+TN+1e-10)
    F1 = stats.hmean([precision, recall])
    print('Eval results: Acc %.2f, precision %.2f, recall %.2f, F1 %.2f, mIoU %.2f.'%(acc*100, precision*100, recall*100, F1*100, mIoU*100))
    
    #Below are the evaluation metrics provided in CTD-Former (https://ieeexplore.ieee.org/document/10139838).
    mIoU = get_mIoU(2, GTs, preds)
