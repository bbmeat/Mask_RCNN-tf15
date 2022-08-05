# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))  # 注意：加mask_rcnn目录
import skimage.io
from mrcnn.config import Config
from datetime import datetime
import mrcnn.model as modellib
from mrcnn import visualize

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "models")


# 配置，同train
class StrawberryConfig(Config):
    NAME = "strawberrys"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 2 + 1
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 768
    MAX_GT_INSTANCES = 100
    RPN_ANCHOR_SCALES = (8 * 7, 16 * 7, 32 * 7, 64 * 7, 128 * 7)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 32
    POST_NMS_ROIS_INFERENCE = 250
    POST_NMS_ROIS_TRAINING = 500
    STEPS_PER_EPOCH = 30
    VALIDATION_STEPS = 5
    LEARNING_RATE = 0.01


class InferenceConfig(StrawberryConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights('../logs/strawberrys20220805T1607/mask_rcnn_strawberrys_0006.h5', by_name=True)  # 注意换成你模型的路径
# model.load_weights('models/shapes20190117T1428/mask_rcnn_shapes_0030.h5', by_name=True) # 注意换成你模型的路径
# model.load_weights('mask_rcnn_coco.h5', by_name=True) # 注意换成你模型的路径

class_names = ['greenstrawberry', 'strawberry']
image = skimage.io.imread('../train_data/val/rgb_2.png')  # 注意事换成你要识别的图片

a = datetime.now()
results = model.detect([image], verbose=1)
b = datetime.now()
print("@@ detect duration", (b - a).seconds, 'second')
r = results[0]
# 画图
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])

visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'],