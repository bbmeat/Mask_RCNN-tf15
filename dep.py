import datetime
import os

import cv2 as cv
import depthai as dai
import tensorflow.compat.v1 as tf
# from utility import *
from detect_mask import Mask
import numpy as np
import random
import time
import sys

# model_h5_path = "./logs/frozen_inference_graph_converted.pb"

width = 640
height = 400
confThreshold = 0.5  # 置信阈值
maskThreshold = 0.3  # Mask 阈值

# video setting

local_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

video_path = "./video/strawberry_" + local_time + ".avi"
fourcc = cv.VideoWriter_fourcc(*'XVID')  # 指定视频视频编解码器格式
out = cv.VideoWriter(video_path, fourcc, 30, (width, height), True)


def printSystemInformation(info):
    m = 1024 * 1024  # MiB
    print(f"Ddr used / total - {info.ddrMemoryUsage.used / m:.2f} / {info.ddrMemoryUsage.total / m:.2f} MiB")
    print(f"Cmx used / total - {info.cmxMemoryUsage.used / m:.2f} / {info.cmxMemoryUsage.total / m:.2f} MiB")
    print(
        f"LeonCss heap used / total - {info.leonCssMemoryUsage.used / m:.2f} / {info.leonCssMemoryUsage.total / m:.2f} MiB")
    print(
        f"LeonMss heap used / total - {info.leonMssMemoryUsage.used / m:.2f} / {info.leonMssMemoryUsage.total / m:.2f} MiB")
    t = info.chipTemperature
    print(
        f"Chip temperature - average: {t.average:.2f}, css: {t.css:.2f}, mss: {t.mss:.2f}, upa: {t.upa:.2f}, dss: {t.dss:.2f}")
    print(
        f"Cpu usage - Leon CSS: {info.leonCssCpuUsage.average * 100:.2f}%, Leon MSS: {info.leonMssCpuUsage.average * 100:.2f} %")
    print("----------------------------------------")


# create pipeline
def createPipeline():
    print("creat pipeline")

    pipeline = dai.Pipeline()

    # rgb camera
    camRgb = pipeline.createColorCamera()
    camRgb.setPreviewSize(640, 400)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Define source and output
    sysLog = pipeline.create(dai.node.SystemLogger)

    # 定义灰度相机和
    monoLeft = pipeline.createMonoCamera()
    monoRight = pipeline.createMonoCamera()
    stero = pipeline.createStereoDepth()
    spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

    lrcheck = True
    subpixel = True

    # 输出队列
    xoutPreview = pipeline.createXLinkOut()
    xoutDepth = pipeline.createXLinkOut()
    xoutSpatialData = pipeline.createXLinkOut()
    xinSpatialConfig = pipeline.createXLinkIn()
    linkOut = pipeline.create(dai.node.XLinkOut)

    xoutPreview.setStreamName("preview")
    xoutDepth.setStreamName("depth")
    xoutSpatialData.setStreamName("spatialData")
    xinSpatialConfig.setStreamName("config")
    linkOut.setStreamName("sysinfo")

    # sedth 节点设置
    stero.initialConfig.setConfidenceThreshold(255)
    stero.setDepthAlign(dai.CameraBoardSocket.RGB)
    stero.setLeftRightCheck(lrcheck)
    stero.setSubpixel(subpixel)

    # 连接左右灰度x深度
    monoLeft.out.link(stero.left)
    monoRight.out.link(stero.right)

    # 空间计算
    spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
    stero.depth.link(spatialLocationCalculator.inputDepth)

    # bgr 输出
    camRgb.preview.link(xoutPreview.input)
    sysLog.setRate(1)

    topLeft = dai.Point2f(0.4, 0.4)
    bottomRight = dai.Point2f(0.8, 0.8)

    spatialLocationCalculator.inputConfig.setWaitForMessage(False)
    config = dai.SpatialLocationCalculatorConfigData()
    config.depthThresholds.lowerThreshold = 100
    config.depthThresholds.upperThreshold = 10000
    config.roi = dai.Rect(topLeft, bottomRight)
    spatialLocationCalculator.initialConfig.addROI(config)
    spatialLocationCalculator.out.link(xoutSpatialData.input)
    xinSpatialConfig.out.link(spatialLocationCalculator.inputConfig)
    sysLog.out.link(linkOut.input)

    return pipeline


pipeline = createPipeline()
with dai.Device(pipeline) as device:
    rgb = device.getOutputQueue('preview', maxSize=1, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=1, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("config")
    # text = TextHelper()
    detect_mask = Mask(device)

    qSysInfo = device.getOutputQueue(name="sysinfo", maxSize=1, blocking=False)

    while True:
        previrew = rgb.get()
        inDepth = depthQueue.get()
        sysInfo = qSysInfo.get()
        # inDepthAvg = spatialCalcQueue.get()

        frame = previrew.getFrame()
        depthFrame = inDepth.getFrame()

        result = detect_mask.detect_with_h5(frame, depthFrame)
        # cv.imshow("preview", frame)
        masked_image = cv.cvtColor(result, cv.COLOR_RGB2BGR)
        cv.imshow("result", masked_image)
        out.write(masked_image)
        printSystemInformation(sysInfo)  # 输出设备信息

        key = cv.waitKey(1)
        if key == ord('q'):
            break
