#!/usr/bin/env python3

import cv2
import depthai as dai
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)


def getFrame(queue):
    # Get frame from queue
    frame = queue.get()
    # Convert frame to OpenCV format and return
    return frame.getCvFrame()


def getMonoCamera(pipeline, isLeft):
    # Configure mono camera
    mono = pipeline.createMonoCamera()

    # Set Camera Resolution
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    if isLeft:
        # Get left camera
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        # Get right camera
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    return mono


def getStereoPair(pipeline, monoLeft, monoRight):
    # Configure stereo pair for depth estimation
    stereo = pipeline.createStereoDepth()

    # Checks occluded pixels and marks them as invalid
    stereo.setLeftRightCheck(True)

    # Configure left and right cameras to work as a stereo pair
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    return stereo


xoutRgb.setStreamName("rgb")
ctrl = dai.CameraControl()

# Properties
camRgb.setPreviewSize(640, 400)
camRgb.setInterleaved(False)
ctrl.setManualExposure(19000, 850)
ctrl.setManualWhiteBalance(3600)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
monoLeft = getMonoCamera(pipeline, isLeft=True)
monoRight = getMonoCamera(pipeline, isLeft=False)
stereo = getStereoPair(pipeline, monoLeft, monoRight)
xoutDisp = pipeline.createXLinkOut()
xoutDisp.setStreamName("disparity")

xoutRectifiedLeft = pipeline.createXLinkOut()
xoutRectifiedLeft.setStreamName("rectifiedLeft")

xoutRectifiedRight = pipeline.createXLinkOut()
xoutRectifiedRight.setStreamName("rectifiedRight")

stereo.disparity.link(xoutDisp.input)
stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
stereo.rectifiedRight.link(xoutRectifiedRight.input)

# Linking
camRgb.preview.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    disparityQueue = device.getOutputQueue(name="disparity",
                                           maxSize=1, blocking=False)
    rectifiedLeftQueue = device.getOutputQueue(name="rectifiedLeft",
                                               maxSize=1, blocking=False)
    rectifiedRightQueue = device.getOutputQueue(name="rectifiedRight",
                                                maxSize=1, blocking=False)

    # 计算一个倍增器的颜色映射视差映射
    disparityMultiplier = 255 / stereo.initialConfig.getMaxDisparity()

    cv2.namedWindow("Stereo Pair")
    sideBySide = False
    while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
        disparity = getFrame(disparityQueue)

        # Colormap视差显示。
        disparity = (disparity *
                     disparityMultiplier).astype(np.uint8)
        disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

        # Get the left and right rectified frame.
        leftFrame = getFrame(rectifiedLeftQueue);
        rightFrame = getFrame(rectifiedRightQueue)

        if sideBySide:
            # Show side by side view.
            imOut = np.hstack((leftFrame, rightFrame))
        else:
            # Show overlapping frames.
            imOut = np.uint8(leftFrame / 2 + rightFrame / 2)
        # Convert to RGB.
        imOut = cv2.cvtColor(imOut, cv2.COLOR_GRAY2RGB)
        # Draw scan line.

        # Draw clicked point.
        imOutL = np.uint8(leftFrame)
        imOutR = np.uint8(rightFrame)
        imOutL = cv2.cvtColor(imOutL, cv2.COLOR_GRAY2RGB)
        imOutR = cv2.cvtColor(imOutR, cv2.COLOR_GRAY2RGB)
        # Retrieve 'bgr' (opencv format) frame
        cv2.imshow("rgb", inRgb.getCvFrame())
        cv2.imshow("Stereo Pair", imOut)
        cv2.imshow("Disparity", disparity)
        cv2.imshow("left", imOutL)
        cv2.imshow("right", imOutR)
        key = cv2.waitKey(1)
        if cv2.waitKey(1) == ord('q'):
            break
        elif key == ord('t'):
            # Toggle display when t is pressed
            sideBySide = not sideBySide