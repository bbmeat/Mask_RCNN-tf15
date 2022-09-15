# -*- coding: utf-8 -*-

import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import time
import HostSpatialsCalc
import depthai as dai

from mrcnn.config import Config
from datetime import datetime

# Root directory of the project
# ROOT_DIR = os.path.abspath("./")
ROOT_DIR = os.getcwd()
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn import model
import congfig
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "train_data/val")

strawberry = congfig.StrawberryConfig()


def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# import train_tongue
# class InferenceConfig(coco.CocoConfig):
class InferenceConfig(strawberry):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False


inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

model_path = "G:/Python/Mask_RCNN-tf15/logs/mask_rcnn_shapes.h5"
# model_path = model.find_last()


# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

class_names = ['BG', 'greenstrawberry', 'strawberry']

image = skimage.io.imread("G:/Python/Mask_RCNN-tf15/train_data/val/rgb.png")
# image = skimage.io.imread("G:/Python/Mask_RCNN-tf15/train_data/val/rgb2.png")
# image = skimage.io.imread("G:/Python/Mask_RCNN-tf15/train_data/val/rgb3.png")

a = datetime.now()
# Run detection
# print([image])
results = model.detect([image], verbose=1)
b = datetime.now()
# Visualize results
print("shijian", (b - a).seconds)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], figsize=(8, 8))
print('roi', r['rois'])
area1 = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], figsize=(8, 8))

