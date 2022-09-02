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
import mask_count

from mrcnn.config import Config
from datetime import datetime

# Root directory of the project
# ROOT_DIR = os.path.abspath("./")
ROOT_DIR = os.getcwd()
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn import visualize

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "train_data/val")


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 768

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 7, 16 * 7, 32 * 7, 64 * 7, 128 * 7)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 30

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# import train_tongue
# class InferenceConfig(coco.CocoConfig):
class InferenceConfig(ShapesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False


inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

model_path = "G:/Python/Mask_RCNN-tf15/logs/shapes20220901T1707/mask_rcnn_shapes_0005.h5"
# model_path = model.find_last()


# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

class_names = ['BG', 'greenstrawberry', 'strawberry']

image = skimage.io.imread("G:/Python/Mask_RCNN-tf15/train_data/val/rgb.png")
# image = skimage.io.imread("G:/Python/Mask_RCNN-tf15/train_data/val/rgb2.png")
# image = skimage.io.imread("G:/Python/Mask_RCNN-tf15/train_data/val/rgb3.png")
# file_names = next(os.walk(IMAGE_DIR))[2]
# image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

a = datetime.now()
# Run detection
results = model.detect([image], verbose=1)
b = datetime.now()
# Visualize results
print("shijian", (b - a).seconds)
r = results[0]
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], figsize=(8, 8))


