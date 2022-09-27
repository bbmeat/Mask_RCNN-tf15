import cv2
import numpy as np
import os
import sys

sys.path.append("G:\\Python\\Mask_RCNN-tf15\\mrcnn")

from mrcnn import *
from mrcnn import visualize

class MaskRcnn:
    def __int__(self):
        # self.model, self.config =loadmodel("../logs/mask_rcnn.h5")

        np.random.seed(2)
        self.colors = np.random.randint(0, 255, (90, 3))

        # Conf threshold
        self.detection_threshold = 0.7
        self.mask_threshold = 0.3

        self.classes = []
        with open("dnn/classes.txt", "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []
        self.obj_contours = []

        # Distances
        self.distances = []