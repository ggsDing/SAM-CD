#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
import math
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw
import utils.joint_transforms as joint_transforms

import cv2

ZUR_COLORMAP = [[0,0,0],[255,255,255],[0,255,0],[150,80,0],[0,125,0],[150,150,255],[100,100,100],[255,255,0],[0,0,150]]
ZUR_CLASSES  = ['road','background','grass','ground','tree','water','building','raiway','river']

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ZUR_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i) for i in range(n) for id in ids)


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_label, crop_size):
    """Return all the couples (img, mask)"""
    data, labels = [], []
    for id, pos in ids:
        fPath = dir_img + id + '.png'
        lPath = dir_label + id + '.png'
        data.append(np.array(Image.open(fPath)))
        labels.append(np.array(Image.open(lPath)))
    print('%d images loaded.'%len(data))

    data, labels = DataAug(data, labels, crop_size)
    print('Image augment done. %d Images created.'%len(data))
    # need to transform from HWC to CHW
    # imgs_switched = map(hwc_to_chw, data)
    imgs_switched = []
    for im in data:
        imgs_switched.append(hwc_to_chw(im))
    # for i in range(data.shape[0]):
    #     print(data[i].shape)
    # imgs_normalized = map(normalize, imgs_switched)

    labels_index = []
    for label in labels:
        labels_index.append(Color2Index0(label, colormap2label))

    return list(zip(imgs_switched, labels_index))

def get_binary_imgs_and_masks(ids, dir_img, dir_label, crop_size):
    """Return all the couples (img, mask)"""
    data, labels = [], []
    for id, pos in ids:
        fPath = dir_img + id + '.png'
        lPath = dir_label + id + '.png'
        data.append(np.array(Image.open(fPath).convert("L")))
        labels.append(np.asarray(Image.open(lPath).convert("L")))
    print('%d images loaded.'%len(data))

    data, labels = DataAug_1C(data, labels, crop_size)
    print('Image augment done. %d Images created.'%len(data))

    imgs_switched = []
    for im in data:
        imgs_switched.append(np.expand_dims(im, 0))

    labels_index = []
    for label in labels:
        labels_index.append(label/255)

    return list(zip(imgs_switched, labels_index))

def read_images(ids, dir_img, dir_label):
    n = len(Img_fileList)
    data, label = [None] * n, [None] * n
    for id, pos in ids:
        fPath = dir_img + id + suffix
        lPath = dir_label + id + suffix
        data[i] = Image.open(fPath)
        label[i] = Image.open(lPath)
    return data, label

def DataAug(data, labels, size):
    crop_imgs = create_crops(data[0], size)
    crop_labels = create_crops(labels[0], size)
    for i in range(1, len(data)):
        crop_imgs = np.concatenate((crop_imgs, create_crops(data[i], size)), axis=0)
        crop_labels = np.concatenate((crop_labels, create_crops(labels[i], size)), axis=0)
    # crop_imgs = []
    # crop_labels = []
    # aug_times = []
    # ten_crop_imgs = []
    # ten_crop_labels = []
    # for i in range(len(data)):
    #     h_rate = data[i].shape[0]/size[0]
    #     w_rate = data[i].shape[1]/size[1]
    #     aug_time = min(h_rate,w_rate)*2
    #     print(aug_time)
    #     if (aug_time<1.5): aug_time=8
    #     elif (aug_time<2): aug_time=10
    #     else: aug_time=18
    #     aug_times.append(aug_time)
    #     ten_crop_imgs.append(ten_crop(data[i], size))
    #     ten_crop_labels.append(ten_crop(labels[i], size))
    # for t in range(max(aug_times)):
    #     for i in range(len(data)):
    #         if(aug_times[i]>t):
    #             crop_imgs.append(ten_crop_imgs[i][t])
    #             crop_labels.append(ten_crop_labels[i][t])
    return crop_imgs, crop_labels

def DataAug_1C(data, labels, size):
    crop_imgs = create_crops_1C(data[0], size)
    crop_labels = create_crops_1C(labels[0], size)
    for i in range(1, len(data)):
        crop_imgs = np.concatenate((crop_imgs, create_crops_1C(data[i], size)), axis=0)
        crop_labels = np.concatenate((crop_labels, create_crops_1C(labels[i], size)), axis=0)
    # crop_imgs = []
    # crop_labels = []
    # aug_times = []
    # ten_crop_imgs = []
    # ten_crop_labels = []
    # for i in range(len(data)):
    #     h_rate = data[i].shape[0]/size[0]
    #     w_rate = data[i].shape[1]/size[1]
    #     aug_time = min(h_rate,w_rate)*2
    #     print(aug_time)
    #     if (aug_time<1.5): aug_time=8
    #     elif (aug_time<2): aug_time=10
    #     else: aug_time=18
    #     aug_times.append(aug_time)
    #     ten_crop_imgs.append(ten_crop(data[i], size))
    #     ten_crop_labels.append(ten_crop(labels[i], size))
    # for t in range(max(aug_times)):
    #     for i in range(len(data)):
    #         if(aug_times[i]>t):
    #             crop_imgs.append(ten_crop_imgs[i][t])
    #             crop_labels.append(ten_crop_labels[i][t])
    return crop_imgs, crop_labels

def Color2Index(ColorLabels, colormap2label):
    IndexLabels = np.zeros(ColorLabels.shape[0],ColorLabels.shape[1],
                           ColorLabels.shape[2], 1)
    for i, data in enumerate(ColorLabels):
        data = data.astype('int32')
        idx = (data[:,:,0] * 256 + data[:,:,1]) * 256 + data[:,:,2]
        IndexLabels[i] = colormap2label[idx]
    return IndexLabels

def Color2Index0(ColorLabel, colormap2label):
    data = ColorLabel.astype('int32')
    idx = (data[:,:,0] * 256 + data[:,:,1]) * 256 + data[:,:,2]
    return colormap2label[idx]

def Index2Color(pred, colormap2label):
    x = pred.astype('int32')
    return colormap2label[x, :]

def ten_crop(src, size):
    """Crop 10 regions from an array.
    This is performed same as:
    http://chainercv.readthedocs.io/en/stable/reference/transforms.html#ten-crop

    This method crops 10 regions. All regions will be in shape
    :obj`size`. These regions consist of 1 center crop and 4 corner
    crops and horizontal flips of them.
    The crops are ordered in this order.
    * center crop
    * top-left crop
    * bottom-right crop
    * top-right crop
    * bottom-left crop
    * center crop (flipped horizontally)
    * top-left crop (flipped horizontally)
    * bottom-left crop (flipped horizontally)
    * top-right crop (flipped horizontally)
    * bottom-right crop (flipped horizontally)

    Parameters
    ----------
    src : Numpy array
        Input image.
    size : tuple
        Tuple of length 2, as (width, height) of the cropped areas.

    Returns
    -------
    mxnet.nd.NDArray
        The cropped images with shape (10, size[1], size[0], C)

    """
    h, w, _ = src.shape
    ow, oh = size

    if h < oh or w < ow:
        raise ValueError(
            "Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))

    # h=int(h)
    # w = int(w)
    # ow = int(ow)
    # oh = int(oh)

    tl = src[0:oh, 0:ow, :]
    bl = src[h - oh:h, 0:ow, :]
    tr = src[0:oh, w - ow:w, :]
    br = src[h - oh:h, w - ow:w, :]
    center = src[(h - oh) // 2:(h + oh) // 2, (w - ow) // 2:(w + ow) // 2, :]

    tl_f = cv2.flip(tl, -1)
    bl_f = cv2.flip(bl, -1)
    tr_f = cv2.flip(tr, -1)
    br_f = cv2.flip(br, -1)
    center_f = cv2.flip(center, -1)
    print(center_f.shape)
    print(center_rf.shape)
    print(center_tf.shape)
    print(center_bf.shape)
    crops = np.stack([tl, br, tr, bl, tl_f, br_f, tr_f, bl_f, center, center_f], axis=0)
    return crops

def create_crops(img, size):
    # print(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    c_h = size[0]
    c_w = size[1]
    if h < c_h or w < c_w:
        raise ValueError(
            "Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))

    h_rate = h/c_h
    w_rate = w/c_w
    h_times = math.ceil(h_rate)
    w_times = math.ceil(w_rate)
    stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))
    stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
    crop_imgs = []
    for j in range(h_times):
        for i in range(w_times):
            s_h = int(j*c_h - j*stride_h)
            if(j==(h_times-1)): s_h = h - c_h
            e_h = s_h + c_h
            s_w = int(i*c_w - i*stride_w)
            if(i==(w_times-1)): s_w = w - c_w
            e_w = s_w + c_w
            # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
            crop_im = img[s_h:e_h, s_w:e_w, :]
            crop_imgs.append(crop_im)

    crop_imgs_f = []
    for im in crop_imgs:
        crop_imgs_f.append(cv2.flip(im, -1))

    crops = np.concatenate((np.array(crop_imgs), np.array(crop_imgs_f)), axis=0)
    # print(crops.shape)
    return crops

def create_crops_1C(img, size):
    # print(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    c_h = size[0]
    c_w = size[1]
    if h < c_h or w < c_w:
        raise ValueError(
            "Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))

    h_rate = h/c_h
    w_rate = w/c_w
    h_times = math.ceil(h_rate)
    w_times = math.ceil(w_rate)
    stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))
    stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
    crop_imgs = []
    for j in range(h_times):
        for i in range(w_times):
            s_h = int(j*c_h - j*stride_h)
            if(j==(h_times-1)): s_h = h - c_h
            e_h = s_h + c_h
            s_w = int(i*c_w - i*stride_w)
            if(i==(w_times-1)): s_w = w - c_w
            e_w = s_w + c_w
            # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
            crop_im = img[s_h:e_h, s_w:e_w]
            crop_imgs.append(crop_im)

    crop_imgs_f = []
    for im in crop_imgs:
        crop_imgs_f.append(cv2.flip(im, -1))

    crops = np.concatenate((np.array(crop_imgs), np.array(crop_imgs_f)), axis=0)
    # print(crops.shape)
    return crops
