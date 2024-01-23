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
import csv
import glob
from regions import Regions
import math
import copy
import time
import pickle
from regions import Regions
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize
import matplotlib.pyplot as plt
from tqdm import tqdm
import pprint
from shapely.geometry import Polygon, Point
from shapely import box
import random
import threshhold_img
import sys
import datetime
import shutil
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import display, clear_output
import matplotlib.gridspec as gridspec

def create_dirs(name):
    """
    Input: The name of the parent folder of the to-be created dataset. This method creates the directory structure based off the dataset name.

    Output: The global directory path
    """
    
    global_dir = os.path.join('DATASET', name)
    ann_train_path = os.path.join(global_dir,'data','my_dataset','ann','train')
    ann_val_path = os.path.join(global_dir,'data','my_dataset','ann','val')
    img_train_path = os.path.join(global_dir,'data','my_dataset','img','train')
    img_val_path = os.path.join(global_dir,'data','my_dataset','img','val')

    if os.path.exists(global_dir):
        print(global_dir + ' exists!')
        proceed = input('Overwrite? [y/n] ')
        if proceed == 'n':
            print('Exiting...')
            return None
        print('Overwriting directories...')
        shutil.rmtree(global_dir)
    print('Creating directories...')
    os.makedirs(global_dir)
    os.makedirs(ann_train_path)
    os.makedirs(ann_val_path)
    os.makedirs(img_train_path)
    os.makedirs(img_val_path)
    return global_dir


def print_orginial_data_stats(dataset, index):
    #  Inspect logged data. These files contain statistics regarding the raw data, before the zscale is applied.
    radio_data = pd.read_csv(os.path.join('DATASET', dataset, 'radio_data.csv'))
    ha_data = pd.read_csv(os.path.join('DATASET', dataset, 'ha_data.csv'))
    
    rad = radio_data.loc[index]
    ha = ha_data.loc[index]
    
    # Concatenate the DataFrames side by side
    combined_data = pd.concat([rad, ha], axis=1, keys=['Radio Data', 'Ha Data'])
    
    print('\nOriginal data statistics\n')
    print(combined_data)

def display_images(dataset, hist=True, index=None, z_scale='og'):
    """
    Input: The name of the dataset. 
           If scale='norm', normalize display image to [0,1]. If scale='zscale', perform zscale
           If hist=True, display histograms below images or each channel.
           If an index is provided, only show that image and then return
    Output: None. Displays data in jupyter notebook
    """
    dr = os.path.join('DATASET',dataset,'data','my_dataset','img','train')
    histrange = (0,255)
    bins=255
    if index is not None:
        idx = index
    else:
        idx = 0
    while True:
        entry = f"{dr}/{idx}.npy"
        print(entry)
        img = np.load(entry)
        z_img = float_to_int(apply_zscale(np.load(entry), z_scale=z_scale), make3channel=True)
        ann = Image.open(get_companion(entry))
        if hist is not None:
            fig = plt.figure(figsize=(16, 16))
            gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
            # Subplot 1 (Top Left) Annotation
            
            ax1 = plt.subplot(gs[0, 0])
            ax1.imshow(ann)
            ax1.set_title('Annotation')

            # Subplot 2 (Top Right) Image
            ax2 = plt.subplot(gs[0, 1])
            ax2.imshow(z_img)
            ax2.set_title('Image')
            
            # Subplot 3 (Middle Left) Radio hist of Image
            ax3 = plt.subplot(gs[1, 0])
            ax3.hist(z_img[:,:,0].ravel(), bins=bins, range=histrange, alpha=0.7)
            ax3.set_title('Radio (Red Channel) Data')
            ax3.set_xlabel('Pixel Value')
            ax3.set_ylabel('Frequency')
            ax3.grid(axis='y', alpha=0.75)
            
            # Subplot 4 (Middle Right) HA hist of image
            ax4 = plt.subplot(gs[1, 1])
            ax4.hist(z_img[:,:,2].ravel(), bins=bins, range=histrange, alpha=0.7)
            ax4.set_title('H-Alpha (Blue Channel) Data')
            ax4.set_xlabel('Pixel Value')
            ax4.set_ylabel('Frequency')
            ax4.grid(axis='y', alpha=0.75)
        
            # Subplot 5 (Bottom Left) Radio hist of original image data
            ax3 = plt.subplot(gs[2, 0])
            ax3.hist(img[:,:,0].ravel(), bins=bins, alpha=0.7)
            ax3.set_title('Radio (Red Channel) ORIGINAL Data')
            ax3.set_xlabel('Pixel Value')
            ax3.set_ylabel('Frequency')
            ax3.grid(axis='y', alpha=0.75)
            
            # Subplot 6 (Bottom Right) HA hist of orginal image data
            ax4 = plt.subplot(gs[2, 1])
            ax4.hist(img[:,:,1].ravel(), bins=bins, alpha=0.7)
            ax4.set_title('H-Alpha (Blue Channel) ORIGINAL Data')
            ax4.set_xlabel('Pixel Value')
            ax4.set_ylabel('Frequency')
            ax4.grid(axis='y', alpha=0.75)
        else:
            fig, axs = plt.subplots(1, 2, figsize=(16,8))
            axs[0].imshow(ann)
            axs[0].set_title('Annotation')
            axs[1].imshow(z_img)
            axs[1].set_title('Image')
        plt.suptitle(f"Image {idx}")
        plt.show() 
        print_orginial_data_stats(dataset, idx)
        if index is not None:
            break
        action = input('Press enter for next image, \'p\' for previous image, or enter image id. \'q\' to quit')
        if action.isdigit():
            idx = int(action)
        elif action == 'p':
            idx = idx-1
        elif action == 'q':
            break
        else:
            idx = idx+1
        clear_output(wait=True)
        plt.close()
    plt.close()
    return


def get_companion(file_path):
    """
    Input: The img or ann file path

    Output: If file path is img/train/xxx.npy, it will find ann/train/xxx.png.
            If file path is ann/val/yyy.npy, it will find img/val/yyy.npy. 
    """
    comp_file = ''
    if 'img' in file_path:
        # File is img. Replace img with ann and .npy with .png
        comp_file = file_path.replace('img', 'ann')
        comp_file = comp_file.replace('npy', 'png')
    else:
        # File is ann. Replace ann with img and png with npy
        comp_file = file_path.replace('ann', 'img')
        comp_file = comp_file.replace('npy', 'png')
    return comp_file


def float_to_int(img, make3channel=False):
    """
    Input: A floating point img. If make3channel=True, the function will convert to a 3 channel image before returning

    Output: The floating point data will be clipped to (0,1) and then scaled to (0,255) and converted to uint8
    """
    new_channels = []
    for channel in range(img.shape[2]):
        channel_data = img[:,:,channel]  # Get data for the current channel
        if np.all(channel_data == 0):
            new_channel = channel_data
        else:
            new_channel = (np.clip(channel_data,0,1) * 255).astype(np.uint8)
        new_channels.append(new_channel)
    ret_img = np.stack(new_channels, axis=-1)
    ret_img = ret_img.astype(np.uint8)
    if make3channel:
        ret_img = get3channel(ret_img)
    return ret_img


def apply_zscale(img, z_scale='og'):
    """
    Input: An img. z_scale='og' for original method, 'alt' for alternate (second) method

    Output: Apply zscale transformation to img
    """

    # Define the Z-Scale normalization interval
    zscale = ZScaleInterval(contrast=0.3)

    # Normalize each channel separately
    
    radio_data = img[:,:,0]  # Get radio data in the first channel
    if np.all(radio_data == 0):
        print('returning un-zscaled image')
        return img

    ha_data = img[:,:,1]
    ha_vmin, ha_vmax = zscale.get_limits(ha_data)  # Compute the Z-Scale limits for h-alpha channel
    ha_norm = ImageNormalize(vmin=ha_vmin, vmax=ha_vmax) # Compute normalization
    if z_scale == 'og':
        #  Perform original zscale method
        radio_vmin, radio_vmax = zscale.get_limits(radio_data)  # Compute the Z-Scale limits for h-alpha channel
    else:
        #  Perform alternate zscale method
        radio_vmin = ha_vmin / 826000  #  Adapt h-alpha limits to radio
        radio_vmax = ha_vmax / 826000
    radio_norm = ImageNormalize(vmin=radio_vmin, vmax=radio_vmax) # Compute normalization with adapted bounds

    # Combine the normalized channels into a single 3-channel image
    return np.stack([radio_norm(radio_data), ha_norm(ha_data)], axis=-1)


def normalize(img):
    """
    Input: An  img, after cleanup_data has been called

    Output: Each channel individually normalized to the range [0,1]
    """
    
    channels = []
    
    for channel in range(img.shape[2]):
        channel_data = img[:,:,channel]
        if np.all(channel_data == 0):
            norm_channel = channel_data
        else:
            norm_channel = (channel_data - np.min(channel_data)) / (np.max(channel_data) - np.min(channel_data))
        channels.append(norm_channel)

    return np.stack(channels, axis=-1)

def cleanup_data(img):
    """
    Input: An img

    Output: The img with all nans and -10000s set to img.min()
    """

    # Define the minimum value (excluding NaNs and -10000s)
    min_value = np.nanmin(img[img != -10000])
    # Replace NaN values and -10000 with the minimum value
    img = np.where(np.logical_or(np.isnan(img), img == -10000), min_value, img)
    return img


def get3channel(image):
    """
    Input: A numpy image
               
    Output: If image is 1 channel: duplicate channel to RGB
            If image is 2 channel: make R0B (green channel is 0)
            If image is 3 channel: return image
    """
    img_dtype = image.dtype

    if image.ndim == 2:
        # Single channel image. Make grayscale
        result_image = np.stack([image,image,image], axis=-1)
    elif image.shape[2] == 2:
        # Two channel. R0B
        result_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
        result_image[:,:,0] = image[:,:,0]
        result_image[:,:,2] = image[:,:,1]
    else:
        # Three or more channels. Return image
        result_image = image
    result_image = result_image.astype(img_dtype)
    return result_image


def merge_images(img_r, img_h):
    """
    Input: Two 1-channel images
    
    Output: One two channel image, with image1 in the red channel and image2 in the blue channel
    """

    # Create a 3-channel numpy array filled with zeros
    result_image = np.zeros((img_r.shape[0], img_r.shape[1], 2), dtype=np.float32)
    # Set the red channel of the result image to be the values from image1
    result_image[:, :, 0] = img_r
    # Set the blue channel of the result image to be the values from image2
    result_image[:, :, 1] = img_h
    result_image = result_image.astype(np.float32)
    return result_image


def gen_img(image_path_r, image_path_h):
    """
    Input: Path to the radio and h-alpha images.

    Output: The combined 2-channel full size image data
    """

    print('Generating image...')
    
    # Halpha image
    img_h = fits.open(image_path_h)[0].data
    #threshhold_img.get_image_statistics(img_h)
    
    # Radio image. Multiply by 1000
    img_r = fits.getdata(image_path_r)[0][0]
    #img_r = img_r * 1000
    #threshhold_img.get_image_statistics(img_r)

    # Merge images and return
    return merge_images(img_r, img_h)



def get_polygons_and_labels(HII_reg_files, SNR_reg_files):
    """
    Input: A list of the HII region files, and a list of the SNR region files.
    
    Output: Returns the list of polygons, and a list of their corresponding labels and bounding boxes.
    """

    print('Generating polygons and labels...')

    # List of shapely Polygon objects
    polygons = []

    # Their corresponding labels
    labels = []

    #  Background is 0

    #  HII Regions -> Label 1
    for file in HII_reg_files:
        curr_region = Regions.read(file, format='ds9')
        Xs = curr_region[0].vertices.x
        Ys = curr_region[0].vertices.y
        polygons.append(Polygon(list(zip(Xs, Ys))))
        labels.append(1)

    #  SNR Regions -> Label 2
    for file in SNR_reg_files:
        curr_region = Regions.read(file, format='ds9')
        Xs = curr_region[0].vertices.x
        Ys = curr_region[0].vertices.y
        polygons.append(Polygon(list(zip(Xs, Ys))))
        labels.append(2)

    return polygons, labels

def correct_bounding_boxes(bounding_boxes, image_shape):
    """
    Corrects bounding boxes to ensure they are within the bounds of the image.

    Parameters:
    - bounding_boxes (list): List of bounding boxes in the format [x_min, y_min, x_max, y_max].
    - image_shape (tuple): Shape of the image array in the format (height, width).

    Returns:
    - corrected_boxes (list): List of corrected bounding boxes.
    """
    
    #print('Correcting bounding boxes...')

    corrected_boxes = []

    for box in bounding_boxes:
        x_min, y_min, x_max, y_max = box

        # Ensure x_min is within bounds
        x_min = max(0, x_min)
        # Ensure y_min is within bounds
        y_min = max(0, y_min)
        # Ensure x_max is within bounds
        x_max = min(image_shape[1], x_max)
        # Ensure y_max is within bounds
        y_max = min(image_shape[0], y_max)

        # Check if the bounding box is valid (non-empty)
        if x_min < x_max and y_min < y_max:
            corrected_boxes.append([x_min, y_min, x_max, y_max])

    return corrected_boxes

def gen_bboxes(polygons, scale_factor=2, shift=True, imgshape=(16740,16740)):
    """
    Input: List of Polygons, scale factor, shift, and image shape for corrections

    Output: The list of (corrected) bounding boxes
    """

    print('Generating bounding boxes...')

    #  Define the list to hold the bounding boxes. If shift=True, creating 5 bounding boxes per polygon
    #  Each bbox is (min_x, min_y, max_x, max_y)
    bboxes = []

    for poly in polygons:
        min_x, min_y, max_x, max_y = poly.bounds
        width = max_x - min_x
        height = max_y - min_y
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        #  Calculate the new larger bounding box
        new_min_x = min_x - width * (scale_factor - 1) / 2
        new_max_x = max_x + width * (scale_factor - 1) / 2
        new_min_y = min_y - height * (scale_factor - 1) / 2
        new_max_y = max_y + height * (scale_factor - 1) / 2
        bboxes.append((int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)))

        if shift:
            #  Create shifted bboxes
            new_width = new_max_x - new_min_x
            new_height = new_max_y - new_min_y

            #  Shift North West
            new_max_x = center_x
            new_min_x = new_max_x - new_width
            new_max_y = center_y
            new_min_y = new_max_y - new_height
            bboxes.append((int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)))

            #  Shift North East
            new_min_x = center_x
            new_max_x = new_min_x + new_width
            new_max_y = center_y
            new_min_y = new_max_y - new_height
            bboxes.append((int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)))

            #  Shift South East
            new_min_x = center_x
            new_max_x = new_min_x + new_width
            new_min_y = center_y
            new_max_y = new_min_y + new_height
            bboxes.append((int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)))

            # Shift South West
            new_max_x = center_x
            new_min_x = new_max_x - new_width
            new_min_y = center_y
            new_max_y = new_min_y + new_height
            bboxes.append((int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)))

    return correct_bounding_boxes(bboxes, imgshape)



def log_array_statistics(array, image_index, global_dir):
    files = [os.path.join(global_dir, 'radio_data.csv'), os.path.join(global_dir, 'ha_data.csv')]
    
    for file in files:
        channel_data = array[:,:,0] if 'radio' in file else array[:,:,1]

        # Calculate statistics
        mean_value = np.mean(channel_data)
        std_deviation = np.std(channel_data)
        min_value = np.min(channel_data)
        percentiles = np.percentile(channel_data, [10, 25, 50, 75, 90, 99])
        max_value = np.max(channel_data)

        # Log the statistics to the CSV file
        with open(file, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            if image_index == 0:
                # Write header only for the first image
                csv_writer.writerow(['Image', 'Mean', 'Std', 'Minimum', '10%', '25%', '50%', '75%', '90%', '99%', 'Maximum'])
            csv_writer.writerow([image_index, mean_value, std_deviation, min_value] + list(percentiles) + [max_value])



