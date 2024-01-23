import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import glob
from regions import Regions
import math
import copy
import time
import pickle
from regions import Regions
from astropy.io import fits
import matplotlib.pyplot as plt
from tqdm import tqdm
import pprint
from shapely.geometry import Polygon, Point
from shapely import box
import random
import threshhold_img
import data_utils
import sys
import datetime
import shutil

global_dir = ""
ann_train_path = ""
ann_val_path = ""
img_train_path = ""
img_val_path = ""

def assign_global_dir_paths(globaldir):
    """
    Input: The global directory for the dataset.
    
    Output: None. Just assigns the paths to the global variables
    """

    global global_dir, ann_train_path, ann_val_path, img_train_path, img_val_path
    
    global_dir = globaldir
    ann_train_path = os.path.join(global_dir,'data','my_dataset','ann','train')
    ann_val_path = os.path.join(global_dir,'data','my_dataset','ann','val')
    img_train_path = os.path.join(global_dir,'data','my_dataset','img','train')
    img_val_path = os.path.join(global_dir,'data','my_dataset','img','val')


def gen_ann(polygons, labels, img_shape):
    """ 
    Input: Polygons and their corresponding labels. Also the image shape
    
    Output: The annotation array and colorized ann
    """

    if os.path.exists('ann.npy'):
        print('Loading the annotation array...')
        ann = np.load('ann.npy')

    else:
        print('Generating the annotation array...')
        ann = np.zeros(img_shape[:2], dtype=np.uint8)
        
        #  Sort polygons largest to smallest by area
        sorted_polygons = sorted(polygons, key=(lambda polygon: polygon.area), reverse=True)
        for i, poly in enumerate(tqdm(sorted_polygons)):
            min_x, min_y, max_x, max_y = poly.bounds
            label = labels[i]

            for x in range(int(min_x), int(max_x) + 1):
                for y in range(int(min_y), int(max_y) + 1):
                    point = Point(x, y)
                    if poly.contains(point):
                        #  Point is either HII or SNR
                        ann[y, x] = label
        
        #  Save ann.npy to file
        file_path = 'ann.npy'
        np.save(file_path, ann)
        print('Saved ' + file_path)

    #  Save ann.png to file
    ground_truth_image = Image.fromarray(ann)
    file_path = 'ann.png'
    ground_truth_image.save(file_path)
    print('Saved ' + file_path)

    return ann


def gen_basesem(img, ann, bboxes, shuffle=None):
    """
    Input: The full img and ann images, and the list of bboxes

    Output: None. Just creates and saves the dataset.
    """

    #  Shuffle if necessary
    if shuffle is not None:
        r = random.Random(shuffle)
        r.shuffle(bboxes)

    #  Divide bboxes into train and val lists
    val_ratio = 0.2
    split_point = int(len(bboxes) * val_ratio)
    train_bboxes = bboxes[split_point:]
    val_bboxes = bboxes[:split_point]
    
    #  Loop training boxes
    print('Looping training bboxes...')
    for i, bbox in enumerate(tqdm(train_bboxes)):
        min_x, min_y, max_x, max_y = bbox
        data = img[min_y:max_y,min_x:max_x,:]
        data = data_utils.cleanup_data(data)
        data_utils.log_array_statistics(data, i, global_dir)
        img_save = os.path.join(img_train_path, str(i)+'.npy')
        np.save(img_save, data)

        cur_ann = Image.fromarray(ann[min_y:max_y,min_x:max_x])
        ann_save = os.path.join(ann_train_path, str(i)+'.png')
        cur_ann.save(ann_save)

    #  Loop validation boxes
    print('Looping validation bboxes...')
    for i, bbox in enumerate(tqdm(val_bboxes)):
        min_x, min_y, max_x, max_y = bbox
        data = img[min_y:max_y,min_x:max_x,:]
        data = data_utils.cleanup_data(data)
        #data_utils.log_array_statistics(data, i, global_dir)
        img_save = os.path.join(img_val_path, str(i)+'.npy')
        np.save(img_save, data)

        cur_ann = Image.fromarray(ann[min_y:max_y,min_x:max_x])
        ann_save = os.path.join(ann_val_path, str(i)+'.png')
        cur_ann.save(ann_save)



def init(dir_name='', shuffle=None, shift=False, scale_factor=3):

    #  Define the path to the full image
    image_path_radio = './LMC/lmc_askap_aconf.fits'
    image_path_halpha = './LMC/lmc_ha_csub.fits'

    stats = 'Dataset name: ' + dir_name
    stats += '\nShuffle: ' + str(shuffle)
    stats += '\nShift: ' + str(shift)
    stats += '\nScale factor: ' + str(scale_factor)
    print(stats)

    #  Define the HII and SNR region files
    HII_reg_files = glob.glob(os.path.join('./LMC/HII_boundaries', '*.reg'))
    SNR_reg_files = glob.glob(os.path.join('./LMC/SNR_boundaries', '*.reg'))
    
    #  Remove bad files
    HII_reg_files.remove('./LMC/HII_boundaries/mcels-l381.reg')
    HII_reg_files.remove('./LMC/HII_boundaries/mcels-l279.reg')

    #  Create dataset directories, and then assign the global directory paths
    glob_dir = data_utils.create_dirs(dir_name)
    if glob_dir:
        assign_global_dir_paths(glob_dir)
    else:
        return

    #  Generate combined image
    img = data_utils.gen_img(image_path_radio, image_path_halpha)
    
    #  Retrieve list of Polygons and corresponding labels
    polygons, labels = data_utils.get_polygons_and_labels(HII_reg_files, SNR_reg_files)
    
    #  Generate list of bounding boxes for dataset generation
    bboxes = data_utils.gen_bboxes(polygons, scale_factor=scale_factor, shift=shift, imgshape=img.shape[:2])

    #  Load or generate/save ann
    ann = gen_ann(polygons, labels, img.shape)

    #  Generate dataset
    gen_basesem(img, ann, bboxes, shuffle=shuffle)
    print('All done yay')




