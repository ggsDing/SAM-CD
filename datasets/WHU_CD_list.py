import os
import math
import random
import numpy as np
from skimage import io, exposure
from torch.utils import data
from skimage.transform import rescale
from torchvision.transforms import functional as F

num_classes = 1
MEAN = np.array([123.675, 116.28, 103.53])
STD  = np.array([58.395, 57.12, 57.375])
root = '/YOUR_DATA_ROOT/'

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def normalize_image(im):
    #im = (im - MEAN) / STD
    im = im/255
    return im.astype(np.float32)

def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs

def Color2Index(ColorLabel):
    IndexMap = ColorLabel.clip(max=1)
    return IndexMap

def Index2Color(pred):
    pred = exposure.rescale_intensity(pred, out_range=np.uint8)
    return pred

def sliding_crop_CD(imgs1, imgs2, labels, size):
    crop_imgs1 = []
    crop_imgs2 = []
    crop_labels = []
    label_dims = len(labels[0].shape)
    for img1, img2, label in zip(imgs1, imgs2, labels):
        h = img1.shape[0]
        w = img1.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            crop_imgs1.append(img1)
            crop_imgs2.append(img2)
            crop_labels.append(label)
            continue
        h_rate = h/c_h
        w_rate = w/c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        if h_times==1: stride_h=0
        else:
            stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))            
        if w_times==1: stride_w=0
        else:
            stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j*c_h - j*stride_h)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i*c_w - i*stride_w)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
                crop_imgs1.append(img1[s_h:e_h, s_w:e_w, :])
                crop_imgs2.append(img2[s_h:e_h, s_w:e_w, :])
                if label_dims==2:
                    crop_labels.append(label[s_h:e_h, s_w:e_w])
                else:
                    crop_labels.append(label[s_h:e_h, s_w:e_w, :])

    print('Sliding crop finished. %d pairs of images created.' %len(crop_imgs1))
    return crop_imgs1, crop_imgs2, crop_labels

def rand_crop_CD(img1, img2, label, size):
    # print(img.shape)
    h = img1.shape[0]
    w = img1.shape[1]
    c_h = size[0]
    c_w = size[1]
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})"
              .format(str(size), h, w))
    else:
        s_h = random.randint(0, h-c_h)
        e_h = s_h + c_h
        s_w = random.randint(0, w-c_w)
        e_w = s_w + c_w

        crop_im1 = img1[s_h:e_h, s_w:e_w, :]
        crop_im2 = img2[s_h:e_h, s_w:e_w, :]
        crop_label = label[s_h:e_h, s_w:e_w]
        # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
        return crop_im1, crop_im2, crop_label

def rand_flip_CD(img1, img2, label):
    r = random.random()
    # showIMG(img.transpose((1, 2, 0)))
    if r < 0.25:
        return img1, img2, label
    elif r < 0.5:
        return np.flip(img1, axis=0).copy(), np.flip(img2, axis=0).copy(), np.flip(label, axis=0).copy()
    elif r < 0.75:
        return np.flip(img1, axis=1).copy(), np.flip(img2, axis=1).copy(), np.flip(label, axis=1).copy()
    else:
        return img1[::-1, ::-1, :].copy(), img2[::-1, ::-1, :].copy(), label[::-1, ::-1].copy()

def read_RSimages(mode, list_name):
    #assert mode in ['train0', 'val0', 'test0']
    img_A_dir = os.path.join(root, 'A')
    img_B_dir = os.path.join(root, 'B')
    label_dir = os.path.join(root, 'label')
    
    if mode=='train':
        list_path=os.path.join(root, 'list', list_name+'.txt')
    else:
        list_path=os.path.join(root, 'list', 'val.txt')
    list_info = open(list_path, 'r')
    data_list = list_info.readlines()
    data_list = [item.rstrip() for item in data_list]
    
    data_A, data_B, labels = [], [], []
    for idx, it in enumerate(data_list):
        if (it[-4:]=='.png'):
            img_A_path = os.path.join(img_A_dir, it)
            img_B_path = os.path.join(img_B_dir, it)
            label_path = os.path.join(label_dir, it)
            
            img_A = io.imread(img_A_path)
            img_A = normalize_image(img_A)
            img_B = io.imread(img_B_path)
            img_B = normalize_image(img_B)
            label = Color2Index(io.imread(label_path))
            
            data_A.append(img_A)
            data_B.append(img_B)
            labels.append(label)
        #if idx>10: break    
        if not idx%100: print('%d/%d images loaded.'%(idx, len(data_list)))
    print(data_A[0].shape)
    print(str(len(data_A)) + ' ' + mode + ' images' + ' loaded.')        
    return data_A, data_B, labels

class RS(data.Dataset):
    def __init__(self, mode, list_name='train', random_crop=False, crop_nums=6, sliding_crop=False, crop_size=512, random_flip=False):
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.crop_nums = crop_nums
        self.crop_size = crop_size
        data_A, data_B, labels = read_RSimages(mode, list_name)
        if sliding_crop:
            data_A, data_B, labels = sliding_crop_CD(data_A, data_B, labels, [self.crop_size, self.crop_size])   
        self.data_A, self.data_B, self.labels = data_A, data_B, labels
        if self.random_crop:
            self.len = crop_nums*len(self.data_A)
        else:
            self.len = len(self.data_A)

    def __getitem__(self, idx):
        if self.random_crop:
            idx = idx//self.crop_nums
        data_A = self.data_A[idx]
        data_B = self.data_B[idx]
        label = self.labels[idx]
        if self.random_crop:
            data_A, data_B, label = rand_crop_CD(data_A, data_B, label, [self.crop_size, self.crop_size])
        if self.random_flip:
            data_A, data_B, label = rand_flip_CD(data_A, data_B, label)
        return F.to_tensor(data_A), F.to_tensor(data_B), label

    def __len__(self):
        return self.len

