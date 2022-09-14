#!/usr/bin/env python3

import cv2 as cv
import depthai as dai
from calc import HostSpatialsCalc
from utility import *
import numpy as np
import math
import test_model

# 给出模型的 textGraph 和权重文件
textGraph = "./Model/mask_rcnn_landing.pbtxt"
modelWeights = "/Model/mask_rcnn_landing.pb."

net = cv.dnn.readNetFromTensorflow(modelWeights, textGraph)

# Create pipeline
pipeline = dai.Pipeline()

# 定义源和输出
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

stereo = pipeline.create(dai.node.StereoDepth)

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo.initialConfig.setConfidenceThreshold(255)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)
stereo.setExtendedDisparity(True)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("disp")
stereo.disparity.link(xoutDepth.input)

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
ctrl = dai.CameraControl()

# Properties
camRgb.setPreviewSize(640, 400)
camRgb.setInterleaved(False)
ctrl.setManualExposure(19000, 850)
ctrl.setManualWhiteBalance(3600)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
camRgb.preview.link(xoutRgb.input)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    cap = cv2.VideoCapture(0)
    # 输出队列将用于从上面定义的输出中获取深度帧
    depthQueue = device.getOutputQueue(name="depth")
    dispQ = device.getOutputQueue(name="disp")
    text = TextHelper()
    hostSpatials = HostSpatialsCalc(device)
    y1, x1, y2, x2 = test_model.r['rois'][0]
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    while True:
        RgbFrame = qRgb.get().getCvFrame()
        frame = cap.read()