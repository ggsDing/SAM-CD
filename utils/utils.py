import os
import random
import numpy as np
from scipy import stats

def read_idtxt(path):
  id_list = []
  #print('start reading')
  f = open(path, 'r')
  curr_str = ''
  while True:
      ch = f.read(1)
      if is_number(ch):
          curr_str+=ch
      else:
          id_list.append(curr_str)
          #print(curr_str)
          curr_str = ''      
      if not ch:
          #print('end reading')
          break
  f.close()
  return id_list

def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)

def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def seprate_batch(dataset, batch_size):
    """Yields lists by batch"""
    num_batch = len(dataset)//batch_size+1
    batch_len = batch_size
    # print (len(data))
    # print (num_batch)
    batches = []
    for i in range(num_batch):
        batches.append([dataset[j] for j in range(batch_len)])
        # print('current data index: %d' %(i*batch_size+batch_len))
        if (i+2==num_batch): batch_len = len(dataset)-(num_batch-1)*batch_size
    return(batches)

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255

def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


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

def ImageValStretch2D(img):
    img = img*255
    #maxval = img.max(axis=0).max(axis=0)
    #minval = img.min(axis=0).min(axis=0)
    #img = (img-minval)*255/(maxval-minval)
    return img.astype(int)

def ConfMap(output, pred):
    # print(output.shape)
    n, h, w = output.shape
    conf = np.zeros(pred.shape, float)
    for h_idx in range(h):
      for w_idx in range(w):
        n_idx = int(pred[h_idx, w_idx])
        sum = 0
        for i in range(n):
          val=output[i, h_idx, w_idx]
          if val>0: sum+=val
        conf[h_idx, w_idx] = output[n_idx, h_idx, w_idx]/sum
        if conf[h_idx, w_idx]<0: conf[h_idx, w_idx]=0
    # print(conf)
    return conf

def accuracy(pred, label):
    valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def align_dims(np_input, expected_dims=2):
    dim_input = len(np_input.shape)
    np_output = np_input
    if dim_input>expected_dims:
        np_output = np_input.squeeze(0)
    elif dim_input<expected_dims:
        np_output = np_input.unsqueeze(0)        
    assert len(np_output.shape) == expected_dims
    return np_output

def binary_accuracy(pred, label):
    pred = align_dims(pred, 2)
    label = align_dims(label, 2)
    pred = (pred>= 0.5)
    label = (label>= 0.5)
    
    TP = float((pred * label).sum())
    FP = float((pred * (1-label)).sum())
    FN = float(((1-pred) * (label)).sum())
    TN = float(((1-pred) * (1-label)).sum())
    precision = TP / (TP+FP+1e-10)
    recall = TP / (TP+FN+1e-10)
    IoU = TP / (TP+FP+FN+1e-10)
    acc = (TP+TN) / (TP+FP+FN+TN)
    F1 = 0
    if acc>0.999 and TP==0:
        precision=1
        recall=1
        IoU=1
    if precision>0 and recall>0:
        F1 = stats.hmean([precision, recall])
    return acc, precision, recall, F1, IoU

def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # imPred += 1
    # imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass+1))
    # print(area_intersection)

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass+1))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass+1))
    area_union = area_pred + area_lab - area_intersection
    # print(area_pred)
    # print(area_lab)
    return (area_intersection, area_union)

def CaclTP(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    # imPred += 1
    # imLab += 1
    # # Remove classes from unlabeled pixels in gt image.
    # # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    TP = imPred * (imPred == imLab)
    (TP_hist, _) = np.histogram(
        TP, bins=numClass, range=(1, numClass+1))
    # print(TP.shape)
    # print(TP_hist)

    # Compute area union:
    (pred_hist, _) = np.histogram(imPred, bins=numClass, range=(1, numClass+1))
    (lab_hist, _) = np.histogram(imLab, bins=numClass, range=(1, numClass+1))
    
    union_hist = pred_hist + lab_hist - TP_hist
    # print(pred_hist)
    # print(lab_hist)
    # precision = TP_hist / (lab_hist + 1e-10) + 1e-10
    # recall = TP_hist / (pred_hist + 1e-10) + 1e-10
    # # print(precision)
    # # print(recall)
    # F1 = [stats.hmean([pre, rec]) for pre, rec in zip(precision, recall)]
    # print(F1)

    # print(area_pred)
    # print(area_lab)

    return (TP_hist, pred_hist, lab_hist, union_hist)