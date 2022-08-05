#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
# 
# 
"""
    这个笔记本展示了如何在你自己的数据集上训练Mask R-CNN。
    为了保持简单，我们使用能够快速训练的形状合成数据集(正方形、三角形和圆形)。
    不过，你仍然需要GPU，因为网络骨干是Resnet101，在CPU上训练太慢了。
    在GPU上，你可以在几分钟内得到不错的结果，在不到一小时内得到好的结果。
    数据集的代码包含在下面。它实时生成图像，因此不需要下载任何数据。
    它可以生成任何大小的图像，所以我们选择小尺寸的图像来更快地训练。
"""
# In[1]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

import warnings

import yaml
from PIL import Image

warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 项目的根目录
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # 找到本地目录

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# ## Configurations

# In[2]:


class StrawberryConfig(Config):
    """
    用于训练玩具形状数据集的配置。
    从基本的Config类派生，并重写特定于玩具形状数据集的值。
    """
    # Give the configuration a recognizable name
    NAME = "strawberrys"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 2 + 1
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 768
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 32
    POST_NMS_ROIS_INFERENCE = 250
    POST_NMS_ROIS_TRAINING = 500
    STEPS_PER_EPOCH = 30
    VALIDATION_STEPS = 5
    LEARNING_RATE = 0.01


config = StrawberryConfig()
config.display()


# ## Notebook Preferences

# In[3]:


def get_ax(rows=1, cols=1, size=8):
    """
    返回一个用于笔记本中的所有可视化的Matplotlib Axes数组。提供一个中心点来控制图形的大小。
    更改默认大小属性以控制渲染图像的大小
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# ## Dataset

# In[4]:
class StrawberryDataset(utils.Dataset):

    # 得到该图像中有多少实例（物体）
    def get_obj_index(self, image):
        n = np.max(image)
        return n

        # 解析labelme中得到的yaml文件，从而得到每个mask对应的实例标签
        # 根据给出的image_id，读取对应的yaml.info文件，其读取内容是字典
        # 取键label_names的值（形式为一个list），去掉labels[0]（也就是背景类）
        # 返回值labels就是一个所有类别名称的list,此处返回值是labels=['hook']

    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.safe_load(f.read())
            labels = temp['label_names']
            # labels = list(labels.keys())
            del labels[0]
        return labels

        # 重新写drcv_img = cv2.imread(dataset_root_path + "labelme_json/" +
        # filestr + "_json/img.png")aw_mask, 根据给出的image_id画出对应的mask
        # 判断image中某点的像素值x，如果x=index+1，说明该像素是第index个目标的mask（像素值=0是背景）
        # 然后将mask[j,i,index]赋值=1，也就是在第index个图上画该像素对应的点
        # 返回值mask是在每个通道index上，已经画好了mask的一系列图片（三维数组）

    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
            return mask

    def load_strawberrys(self, count, height, width, img_floder, mask_floder, imglist, dataset_root_path):
        """生成所请求的合成图像数量。
            count:生成图像的数量。
            height, width:生成图像的大小。
        """
        # Add classes
        self.add_class("strawberrys", 1, "strawberry")
        self.add_class("strawberrys", 2, "greenstrawberry")
        for i in range(count):
            filestr = imglist[i].split(".")[0]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "/labelme_json/" + filestr + ".yaml"
            # if not os.path.exists(mask_path) or not os.path.exists(yaml_path):
            #     continue
            # cv_img = cv2.imread(img_floder + "/" + filestr + ".png")
            # if cv_img is None:
            #     cv_img = cv2.imread(img_floder + "/" + filestr + ".jpg")
            self.add_image("strawberrys",
                           image_id=i,
                           path=img_floder + "/" + imglist[i],
                           width=width, height=height,
                           mask_path=mask_path, yaml_path=yaml_path)

    # def image_reference(self, image_id):
    #     """Return the shapes data of the image."""
    #     info = self.image_info[image_id]
    #     if info["source"] == "strawberrys":
    #         return info["path"]
    #     else:
    #         super(self.__class__).image_reference(self, image_id)

    # 重写load_mask

    def load_mask(self, image_id):
        """为给定图像ID的形状生成实例掩码。
        """
        global iter_num
        info = self.image_info[image_id]
        count = 1  # 检测目标共有1类
        img = Image.open(info['mask_path'])  # 根据mask路径打开图片的mask文件
        num_obj = self.get_obj_index(img)  # 由于mask的规则：第i个目标的mask像素值=i，所以通过像素值最大值，可以知道有多少个目标
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)  # 根据h,w和num创建三维数组（多张mask）
        mask = self.draw_mask(num_obj, mask, img, image_id)  # 调用draw_mask画出mask

        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        # 获取obj_class的列表，此处labels=['hook']
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("strawberry") != -1:
                # print "strawberry"
                labels_form.append("strawberry")
            elif labels[i].find("greenstrawberry") != -1:
                # print "greenstrawberry"
                labels_form.append("greenstrawberry")
        # 生成class_id，其实际上使用class_names中映射过来的
        # 从class_names中找到hook对应的index，然后添加到class_ids中
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)

    # def draw_shape(self, image, shape, dims, color):
    #     """Draws a shape from the given specs."""
    #     # Get the center x, y and the size s
    #     x, y, s = dims
    #     if shape == 'square':
    #         cv2.rectangle(image, (x - s, y - s), (x + s, y + s), color, -1)
    #     elif shape == "circle":
    #         cv2.circle(image, (x, y), s, color, -1)
    #     elif shape == "triangle":
    #         points = np.array([[(x, y - s),
    #                             (x - s / math.sin(math.radians(60)), y + s),
    #                             (x + s / math.sin(math.radians(60)), y + s),
    #                             ]], dtype=np.int32)
    #         cv2.fillPoly(image, points, color)
    #     return image

    # def random_shape(self, height, width):
    #     """生成位于给定高度和宽度边界内的随机形状的规范。
    #     返回一个三值元组:
    #         *形状名称(方形，圆形，…)
    #         *形状颜色:3个值的元组，RGB。
    #         *形状尺寸:一个值的元组，定义形状大小和位置。不同的形状类型。
    #     """
    #     # Shape
    #     shape = random.choice(["square", "circle", "triangle"])
    #     # Color
    #     color = tuple([random.randint(0, 255) for _ in range(3)])
    #     # Center x, y
    #     buffer = 20
    #     y = random.randint(buffer, height - buffer - 1)
    #     x = random.randint(buffer, width - buffer - 1)
    #     # Size
    #     s = random.randint(buffer, height // 4)
    #     return shape, color, (x, y, s)

    # def random_image(self, height, width):
    #     """
    #     创建具有多个形状的图像的随机规范。
    #         返回图像的背景颜色和可用于绘制图像的形状规范列表。
    #     """
    #     # Pick random background color
    #     bg_color = np.array([random.randint(0, 255) for _ in range(3)])
    #     # 生成一些随机的形状并记录它们
    #     # bounding boxes
    #     shapes = []
    #     boxes = []
    #     N = random.randint(1, 4)
    #     for _ in range(N):
    #         shape, color, dims = self.random_shape(height, width)
    #         shapes.append((shape, color, dims))
    #         x, y, s = dims
    #         boxes.append([y - s, x - s, y + s, x + s])
    #     # 使用0.3阈值的非最大抑制来避免形状相互覆盖
    #     keep_ixs = utils.non_max_suppression(np.array(boxes), np.arange(N), 0.3)
    #     shapes = [s for i, s in enumerate(shapes) if i in keep_ixs]
    #     return bg_color, shapes


# In[5]:


# 基础设置
dataset_root_path = "../train_data/"
img_floder = dataset_root_path + "pic"
mask_floder = dataset_root_path + "mask"
yaml_floder = dataset_root_path + "labelme_json"
imglist = os.listdir(img_floder)
count = len(imglist)
width = 640
heigt = 400

# Training dataset
dataset_train = StrawberryDataset()
dataset_train.load_strawberrys(count, 400, 640, img_floder, mask_floder, imglist, dataset_root_path)
dataset_train.prepare()

# Validation dataset
dataset_val = StrawberryDataset()
dataset_val.load_strawberrys(count, 400, 640, img_floder, mask_floder, imglist, dataset_root_path)
dataset_val.prepare()

# In[6]:


# 加载和显示随机样本
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# ## Create Model

# In[17]:


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# In[18]:


# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

# ## Training
# 
# Train in two stages:
# 1. 只有正面。在这里，我们冻结了所有的骨干层，只训练随机初始化的层(即那些我们没有使用MS COCO预训练权重的层)。
# 要只训练头部图层，将' layers='heads'传递给' train() '函数。
#
# 2。调整所有层。对于这个简单的示例，没有必要这样做，但我们将它包括进来以展示整个过程。
# 简单地通过' layers= ' all '来训练所有层。
# In[19]:


# Train the head branches
# 传递 layers="heads" 冻结除头部层以外的所有层。
# 您还可以通过一个正则表达式来根据名称模式选择要训练的层。
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=1,
#             layers='heads')

# In[20]:


# 微调所有图层
# 传递层=“all”训练所有层。您还可以通过一个正则表达式来根据名称模式选择要训练的层。
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2,
            layers="all")

# In[21]:


# Save weights
# 通常不需要，因为回调会在每个epoch之后保存
# 取消注释以手动保存
model_path = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")
model.keras_model.save_weights(model_path)


# ## 检测

# In[22]:


class InferenceConfig(StrawberryConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# In[34]:


# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
                                                                                   image_id)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))

# In[35]:


results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())

# ## Evaluation

# In[25]:


# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
                                                                              image_id)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)

print("mAP: ", np.mean(APs))

# In[ ]:


# In[ ]:
