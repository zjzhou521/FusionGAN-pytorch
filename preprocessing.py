# coding: utf-8
import cv2
import os
import glob
import numpy as np


def im2double(im):
    return im.astype(np.float) / 127.5 - 1


def rgb2ihs(im, eps=2.2204e-16):
    R = im[:, :, 0]
    G = im[:, :, 1]
    B = im[:, :, 2]

    I = (R + G + B) / 3.0
    v1 = (-np.sqrt(2) * R - np.sqrt(2) * G + 2 * np.sqrt(2) * B) / 6.0
    v2 = (R - G) / np.sqrt(2)
    H = np.arctan(v1 / (v2 + eps))
    S = np.sqrt(v1 ** 2 + v2 ** 2)
    # IHS = np.zeros(im.shape)
    # IHS[:,:,0] = I
    # IHS[:,:,1] = H
    # IHS[:,:,2] = S
    return I, v1, v2


def ihs2rgb(im, v1, v2):
    I = im[:, :, 0]
    R = I - v1 / np.sqrt(2) + v2 / np.sqrt(2)
    G = I - v1 / np.sqrt(2) - v2 / np.sqrt(2)
    B = I + np.sqrt(2) * v1
    RGB = np.zeros(im.shape)
    RGB[:, :, 0] = R
    RGB[:, :, 1] = G
    RGB[:, :, 2] = B
    return (RGB)


def prepare_data(data_path):
    """
    Args:
      data_path: choose train dataset or test dataset
      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    data_path = os.path.join(os.getcwd(), data_path) # get absolute path
    images_path = glob.glob(os.path.join(data_path, "*.bmp"))# get a list of imgs
    images_path.extend(glob.glob(os.path.join(data_path, "*.tif")))
    if len(images_path)==0:
        images_path = glob.glob(os.path.join(data_path, "*.jpg"))
    # print(images_path)
    images_path.sort(key=lambda x: int(x[len(data_path) + len(os.path.sep):-4]))# sort by serial
    print('images_path len = ',len(images_path))
    return images_path


# image_size = 132, label_size = 120, stride = 14
def get_images2(data_dir, image_size, label_size, stride):
    data = prepare_data(data_dir) # list of abosulute path
    sub_input_sequence = []
    sub_label_sequence = []
    padding = abs(image_size - label_size) // 2 # 6
    for i in range(len(data)): # for each img path
        input_ = imread(data[i])
        input_ = (input_ - 127.5) / 127.5 # norm [0,255] to [-1,1]
        # print('input_ shape = ',input_.shape) # [height,width]
        height, width = input_.shape[:2]
        # get image patches
        for x in range(0, height - image_size + 1, stride):
            for y in range(0, width - image_size + 1, stride):
                sub_input = input_[x:x + image_size, y:y + image_size].reshape([image_size, image_size, 1])
                sub_label = input_[x + padding:x + padding + label_size, y + padding:y + padding + label_size].reshape(
                    [label_size, label_size, 1])
                sub_input_sequence.append(sub_input)
                sub_label_sequence.append(sub_label)
    sub_input_sequence = np.asarray(sub_input_sequence, dtype=np.float32) # (37710, 132, 132, 1)
    sub_label_sequence = np.asarray(sub_label_sequence, dtype=np.float32) # (37710, 120, 120, 1)
    return sub_input_sequence, sub_label_sequence


def imread(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img[:, :, 0]
