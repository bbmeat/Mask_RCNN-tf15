#!/usr/bin/env python3

import cv2 as cv
import depthai as dai
from utility import *
import numpy as np
import colorsys
import random

confThreshold = 0.5  # 置信阈值
maskThreshold = 0.3  # Mask 阈值
# 给出模型的 textGraph 和权重文件
textGraph = "./Model/mask_rcnn_landing.pbtxt"
modelWeights = "./Model/mask_rcnn_landing.pb"

net = cv.dnn.readNetFromTensorflow(modelWeights, textGraph)

# Create pipeline
pipeline = dai.Pipeline()

# 定义源和输出
camRgb = pipeline.create(dai.node.ColorCamera)
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
classesFile = "mscoco_labels.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
    print("c", classes)

winName = 'Mask-RCNN Object detection '
cv.namedWindow(winName, cv.WINDOW_NORMAL)
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def drawBox(frame, classId, conf, left, top, right, bottom, classMask, area, colors=None):
    # 绘制边界框.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    N = boxes.shape[0]
    colors = colors or random_colors(N)
    # 打印类别标签.
    label = '%.2f' % conf
    if classes:
        assert (classId < len(classes))
        caption = "{} {:.3f}\narea:{:.1f}".format(classes[classId], label, area) if label else conf
    # 在边界框顶部显示标签
    labelSize, baseLine = cv.getTextSize(caption, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    # 调整mask、阈值、颜色并将其应用于图像
    classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
    mask = (classMask > maskThreshold)
    roi = frame[top:bottom + 1, left:right + 1][mask]

    # color = colors[classId%len(colors)]
    # 注释上面的行并取消注释下面的两行以生成不同的实例颜色
    colorIndex = random.randint(0, len(colors) - 1)
    color = colors[colorIndex]

    frame[top:bottom + 1, left:right + 1][mask] = ([0.3 * color[0], 0.3 * color[1], 0.3 * color[2]] + 0.7 * roi).astype(
        np.uint8)

    # 在图像上绘制轮廓
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame[top:bottom + 1, left:right + 1], contours, -1, color, 3, cv.LINE_8, hierarchy, 100)


# 对于每一帧，为每个检测到的对象提取边界框和mask
def postprocess(boxes, masks):
    # mask的输出大小为 NxCxHxW，其中
    # N - 检测到的边界框数量
    # C - 类别数（不包括背景）
    # HxW - 分割形状
    numClasses = masks.shape[1]
    numDetections = boxes.shape[2]

    frameH = RgbFrame.shape[0]
    frameW = RgbFrame.shape[1]

    for i in range(numDetections):
        pre_masks = np.reshape(masks > .5, (-1, masks.shape[-1])).astype(np.float32)
        # 计算mask_面积
        area1 = np.sum(pre_masks, axis=0)
        box = boxes[0, 0, i]
        mask = masks[i]
        area = area1[i]
        score = box[2]
        if score > confThreshold:
            classId = int(box[1])

            # 提取边界框
            left = int(frameW * box[3])
            top = int(frameH * box[4])
            right = int(frameW * box[5])
            bottom = int(frameH * box[6])

            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))

            # 提取对象的mask
            classMask = mask[classId]

            # 绘制边界框，着色并在图像上显示mask
            drawBox(RgbFrame, classId, score, left, top, right, bottom, classMask, area)


# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # cap = cv2.VideoCapture(0)
    # 输出队列将用于从上面定义的输出中获取深度帧
    depthQueue = device.getOutputQueue(name="depth")
    dispQ = device.getOutputQueue(name="disp")
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    while True:
        RgbFrame = qRgb.get().getCvFrame()
        blob = cv.dnn.blobFromImage(RgbFrame, swapRB=True, crop=False)
        net.setInput(blob)
        boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
        postprocess(boxes, masks)
        t, _ = net.getPerfProfile() # 放置效率信息。
        label = 'Mask-RCNN on 2.5 GHz Intel Core i7 CPU, Inference time for a frame : %0.0f ms' % abs(
            t * 1000.0 / cv.getTickFrequency())
        cv.putText(RgbFrame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        cv.imshow(winName, RgbFrame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break