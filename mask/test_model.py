# -*- coding: utf-8 -*-

import os
import sys
import skimage.io
import matplotlib.pyplot as plt

from mask.mrcnn.config import Config
from datetime import datetime

# Root directory of the project
# ROOT_DIR = os.path.abspath("./")
ROOT_DIR = os.getcwd()
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mask.mrcnn.model as modellib
from mask.mrcnn import visualize

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = os.path.join(ROOT_DIR, "train_data/val")

class StrawberryConfig(Config):
    """
    用于训练玩具形状数据集的配置。
    从基本的Config类派生，并重写特定于玩具形状数据集的值。
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 768
    MAX_GT_INSTANCES = 100
    RPN_ANCHOR_SCALES = (8 * 7, 16 * 7, 32 * 7, 64 * 7, 128 * 7)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 100
    POST_NMS_ROIS_INFERENCE = 250
    POST_NMS_ROIS_TRAINING = 500
    STEPS_PER_EPOCH = 30
    VALIDATION_STEPS = 5
    LEARNING_RATE = 0.01

def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# import train_tongue
# class InferenceConfig(coco.CocoConfig):
class InferenceConfig(StrawberryConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False


inference_config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)

model_path = "/mask/logs/mask_rcnn_shapes.h5"
# model_path = model.find_last()


# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

class_names = ['BG', 'strawberry']

image = skimage.io.imread("G:/Python/Mask_RCNN-tf15/train_data/val/rgb.png")
# image = skimage.io.imread("G:/Python/Mask_RCNN-tf15/train_data/val/rgb_12.png")
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

