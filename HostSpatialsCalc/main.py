#!/usr/bin/env python3

import cv2
import depthai as dai
from calc import HostSpatialsCalc
from utility import *
import numpy as np
import math
import test_model

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
stereo.setLeftRightCheck(False)
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
    # 输出队列将用于从上面定义的输出中获取深度帧
    depthQueue = device.getOutputQueue(name="depth")
    dispQ = device.getOutputQueue(name="disp")
    text = TextHelper()
    hostSpatials = HostSpatialsCalc(device)
    y1, x1, y2, x2 = test_model.r['rois'][0]
    x = int((x1 + x2) / 2)
    y = int((y1 + y2))
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    # print("Use WASD keys to move ROI.\nUse 'r' and 'f' to change ROI size.")

    while True:
        depthFrame = depthQueue.get().getFrame()
        # 从深度框架计算空间坐标
        spatials, centroid = hostSpatials.calc_spatials(depthFrame, (x1, y1, x2, y2))
        # 在我们的例子中，Centroid == x/y ，返回值，空间的，形心

        # 获得视差帧以获得更好的深度可视化
        disp = dispQ.get().getFrame()
        disp = (disp * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
        disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
        inRgb = qRgb.get()
        RgbFrame = inRgb.getCvFrame()

        text.rectangle(RgbFrame, (x1, y1), (x2, y2))
        text.putText(RgbFrame, "Height: " + ("{:.1f}cm".format(spatials['h']/10)), (x1 + 10, y1 - 50))
        text.putText(RgbFrame, "a: " + ("{:.1f}cm^2".format(spatials['a']/100)), (x1 + 10, y1 - 30))
        text.putText(RgbFrame, "Z: " + ("{:.1f}cm".format(spatials['z']/10)), (x1 + 10, y1 - 10))
        # text.putText(disp, "Height" + ("{:.1f}mm".format(spatials['h'])), (x1 + 10, y1 - 10))

        # Show the frame
        cv2.imshow("depth", disp)
        cv2.imshow("rgb", RgbFrame)
        cv2.imshow("r", inRgb.getCvFrame())
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
