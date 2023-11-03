import os
import numpy as np
from skimage import io, measure
from scipy import stats
from metric_tool import get_mIoU

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

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

def binary_accuracy(pred, label):
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)
    pred = (pred>= 0.5)
    label = (label>= 0.5)
    
    GT = float((label).sum())
    TP = float((pred * label).sum())
    FP = float((pred * (1-label)).sum())
    FN = float(((1-pred) * (label)).sum())
    TN = float(((1-pred) * (1-label)).sum())
    precision = TP / (TP+FP+1e-10)
    recall = TP / (TP+FN+1e-10)
    IoU = TP / (TP+FP+FN+1e-10)
    acc = (TP+TN) / (TP+FP+FN+TN+1e-10)
    far = (FP)/(TP+FP+1e-10)
    mr = 1-TP/(GT+1e-10)
    F1 = 0
    if acc>0.99 and TP==0:
        precision=1
        recall=1 
        IoU=1       
    if precision>0 and recall>0:
        F1 = stats.hmean([precision, recall])
    return acc, precision, recall, F1, IoU, far, mr
import math

if __name__ == '__main__':
    GT_dir = '/.../levir_CD/label/'
    pred_dir = '/.../eval/Levir_CD/SAM_CD/'
        
    #GT_dir = '/.../WHU_CD/test/label/'
    #pred_dir = '/.../eval/WHU_CD/SAM_CD/'
      
    info_txt_path = os.path.join(pred_dir, 'info.txt')
    f = open(info_txt_path, 'w+')
    acc_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    FAR_meter = AverageMeter()
    MR_meter = AverageMeter()
    
    data_list = os.listdir(pred_dir)
    num_valid = 0
    for idx, it in enumerate(data_list):
        if it[-4:]=='.png': num_valid+=1
    
    preds = []    
    GTs = []
    for idx, it in enumerate(data_list):
        if it[-4:]=='.png':
            pred_path = os.path.join(pred_dir, it)
            pred = io.imread(pred_path)
            h,w = pred.shape[:2]
            #pred = pred[:,:,0]/255.0
            GT_path = os.path.join(GT_dir, it[:-4]+'.png')
            GT = io.imread(GT_path)
            pred = pred//255
            GT = GT//255
            preds.append(pred)
            GTs.append(GT)
            print(GT_path)
            
            #acc, precision, recall, F1, IoU = binary_accuracy(pred, GT)
            acc, precision, recall, F1, IoU, FAR, MR = binary_accuracy(pred, GT[:h,:w])
            acc_meter.update(acc)
            precision_meter.update(precision)
            recall_meter.update(recall)
            F1_meter.update(F1)
            IoU_meter.update(IoU)
            FAR_meter.update(FAR)
            MR_meter.update(MR)
            
            print('Eval num %d/%d, Acc %.2f, precision %.2f, recall %.2f'\
            %(idx, num_valid, acc*100, precision*100, recall*100))
            f.write('Eval num %d/%d, Acc %.2f, precision %.2f, recall %.2f \n'\
            %(idx, num_valid, acc*100, precision*100, recall*100))
    
    mIoU = get_mIoU(2, GTs, preds)