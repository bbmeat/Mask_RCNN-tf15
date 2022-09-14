#!/usr/bin/env python3

import cv2
import depthai as dai
from calc import HostSpatialsCalc
from utility import *
import numpy as np
import math

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
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

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # 输出队列将用于从上面定义的输出中获取深度帧
    depthQueue = device.getOutputQueue(name="depth")
    dispQ = device.getOutputQueue(name="disp")

    text = TextHelper()
    hostSpatials = HostSpatialsCalc(device)
    y = 200
    x = 300
    step = 3
    delta = 5
    hostSpatials.setDeltaRoi(delta)

    print("Use WASD keys to move ROI.\nUse 'r' and 'f' to change ROI size.")

    while True:
        depthFrame = depthQueue.get().getFrame()
        # 从深度框架计算空间坐标
        spatials, centroid = hostSpatials.calc_spatials(depthFrame, (x,y)) # centroid == x/y in our case

        # 获得视差帧以获得更好的深度可视化
        disp = dispQ.get().getFrame()
        disp = (disp * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
        disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        text.rectangle(disp, (x-delta, y-delta), (x+delta, y+delta))
        text.putText(disp, "X: " + ("{:.1f}mm".format(spatials['x']) if not math.isnan(spatials['x']) else "--"), (x + 10, y + 20))
        text.putText(disp, "Y: " + ("{:.1f}mm".format(spatials['y']) if not math.isnan(spatials['y']) else "--"), (x + 10, y + 35))
        text.putText(disp, "Z: " + ("{:.1f}mm".format(spatials['z']) if not math.isnan(spatials['z']) else "--"), (x + 10, y + 50))

        # Show the frame
        cv2.imshow("depth", disp)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('w'):
            y -= step
        elif key == ord('a'):
            x -= step
        elif key == ord('s'):
            y += step
        elif key == ord('d'):
            x += step
        elif key == ord('r'): # Increase Delta
            if delta < 50:
                delta += 1
                hostSpatials.setDeltaRoi(delta)
        elif key == ord('f'): # Decrease Delta
            if 3 < delta:
                delta -= 1
                hostSpatials.setDeltaRoi(delta)