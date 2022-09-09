# Copyright (C) 2018-2019, BigVision LLC (LearnOpenCV.com), All Rights Reserved. 
# Author : Sunita Nayak
# Article : https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/
# License: BSD-3-Clause-Attribution (Please read the license file.)
# This work is based on OpenCV samples code (https://opencv.org/license.html)    

import cv2 as cv
import argparse
import numpy as np
import os.path
import sys
import random

# Initialize the parameters
confThreshold = 0.5  # 置信阈值
maskThreshold = 0.3  # Mask 阈值

parser = argparse.ArgumentParser(description='Use this script to run Mask-RCNN object detection and segmentation')
parser.add_argument('--image', help='Path to image file')
parser.add_argument('--video', help='Path to video file.', default="cars.mp4")
parser.add_argument("--device", default="gpu", help="Device to inference on")
args = parser.parse_args()


# 绘制预测的边界框，着色并在图像上显示mask
def drawBox(frame, classId, conf, left, top, right, bottom, classMask):
    # 绘制边界框.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    # 打印类别标签.
    label = '%.2f' % conf
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # 在边界框顶部显示标签
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
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

    frameH = frame.shape[0]
    frameW = frame.shape[1]

    for i in range(numDetections):
        box = boxes[0, 0, i]
        mask = masks[i]
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
            drawBox(frame, classId, score, left, top, right, bottom, classMask)


# 加载类名
classesFile = "mscoco_labels.names";
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
# 加载颜色
colorsFile = "colors.txt";
with open(colorsFile, 'rt') as f:
    colorsStr = f.read().rstrip('\n').split('\n')
colors = []
for i in range(len(colorsStr)):
    rgb = colorsStr[i].split(' ')
    color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])

# 给出模型的 textGraph 和权重文件
textGraph = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
modelWeights = "./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";

# 加载网络
net = cv.dnn.readNetFromTensorflow(modelWeights, textGraph);

if args.device == "cpu":
    net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

# 从文件中加载颜色
colorsFile = "colors.txt";
with open(colorsFile, 'rt') as f:
    colorsStr = f.read().rstrip('\n').split('\n')
colors = []  # [0,0,0]
for i in range(len(colorsStr)):
    rgb = colorsStr[i].split(' ')
    color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
    colors.append(color)

winName = 'Mask-RCNN Object detection and Segmentation in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "mask_rcnn_out_py.avi"
if (args.image):
    # 打开图像文件
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4] + '_mask_rcnn_out_py.jpg'
elif (args.video):
    # 打开视频文件
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4] + '_mask_rcnn_out_py.avi'
else:
    # 网络摄像头输入
    cap = cv.VideoCapture(0)

# 初始化视频编写器以保存输出视频
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 28,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:

    # 从视频中获取帧
    hasFrame, frame = cap.read()

    # 如果到达视频结尾，则停止程序
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        break

    # 从一帧图像创建一个 4D blob。
    blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)

    # 设置网络输入
    net.setInput(blob)

    # 运行前向传播以从输出层获取输出
    boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

    # 为每个检测到的对象提取边界框和掩码
    postprocess(boxes, masks)

    # 放置效率信息。
    t, _ = net.getPerfProfile()
    label = 'Mask-RCNN on 2.5 GHz Intel Core i7 CPU, Inference time for a frame : %0.0f ms' % abs(
        t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # 用检测框写入帧图像中
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8));
    else:
        vid_writer.write(frame.astype(np.uint8))

    cv.imshow(winName, frame)
