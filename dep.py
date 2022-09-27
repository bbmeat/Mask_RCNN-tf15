import cv2 as cv
import depthai as dai
import tensorflow.compat.v1 as tf
from utility import *
from detect_mask import Mask
import numpy as np
import random

# model_h5_path = "./logs/frozen_inference_graph_converted.pb"
model_path = "./Model/frozen_inference_graph.pb";
model_pbTXT = "./Model/mask_rcnn.pbtxt"
net = cv.dnn.readNetFromTensorflow(model_path, model_pbTXT)
width = 640
height = 400
confThreshold = 0.5  # 置信阈值
maskThreshold = 0.3  # Mask 阈值


# create pipeline

print("creat pipeline")

pipeline = dai.Pipeline()

# rgb camera
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(640, 400)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# 定义灰度相机和
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stero = pipeline.createStereoDepth()
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

lrcheck = True
subpixel = False

    # 输出队列
xoutPreview = pipeline.createXLinkOut()
xoutDepth = pipeline.createXLinkOut()
xoutSpatialData = pipeline.createXLinkOut()
xinSpatialConfig = pipeline.createXLinkIn()

xoutPreview.setStreamName("preview")
xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialConfig.setStreamName("config")

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

"""

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
    print("c",classes)
# 加载颜色
colorsFile = "colors.txt";
with open(colorsFile, 'rt') as f:
    colorsStr = f.read().rstrip('\n').split('\n')
colors = []  # [0,0,0]
for i in range(len(colorsStr)):
    rgb = colorsStr[i].split(' ')
    color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
    colors.append(color)

# 加载TF模型
detection_graph = tf.Graph()
with detection_graph.as_default():
    graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(model_path, 'rb') as f:
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    sess = tf.Session(graph=detection_graph, config=None)

# 输入张量是图像
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# 输出张量是检测框、分数和类
# 每个方框代表图像中检测到特定物体的部分
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# 每个分数代表对每个对象的信心水平。
# 分数显示在结果图像上，以及班级标签。
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
detection_masks = detection_graph.get_tensor_by_name('detection_masks:0')

"""
with dai.Device(pipeline) as device:

    rgb = device.getOutputQueue('preview', maxSize=1, blocking=True)
    depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=True)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=1, blocking=True)
    spatialCalcConfigInQueue = device.getInputQueue("config")
    text = TextHelper()
    detect_mask = Mask(device)

    while True:
        previrew = rgb.get()

        inDepth = depthQueue.get()
        inDepthAvg = spatialCalcQueue.get()

        frame = previrew.getFrame()

        depthFrame = inDepth.getFrame()
        # color_img = np.expand_dims(frame, axis=0)
        # scaled_size = (int(width), int(height))
        # boxes, scores, classes, num, masks = sess.run([detection_boxes, detection_scores, detection_classes,
        #                                                num_detections, detection_masks],
        #                                               feed_dict={image_tensor: color_img})
        # boxes = np.squeeze(boxes)
        # classes = np.squeeze(classes).astype(np.int32)
        # scores = np.squeeze(scores)
        # masks = np.squeeze(masks)
        # print('mask', masks)
        # class_name = ('strawberry')
        # count = 0
        # n = int(num)
        # if not n:
        #     print("\n*** No instances to display *** \n")
        # else:
        #     assert n == int(num)
        #
        # for N in range(n):
        #     class_ = classes[N]
        #     score = scores[N]
        #     box = boxes[N]
        #     mask = masks[N]
        #     print('box', box)
        #     print('cla', class_, 'N', N, )
        #     if score > 0.5 and class_ == 1:
        #         count = count + 1
        #         left = box[1] * width
        #         top = box[0] * height
        #         right = box[3] * width
        #         bottom = box[2] * height
        #
        #         W = right - left
        #         H = bottom - top
        #         bbox = (int(left), int(top), int(width), int(height))
        #         p1 = (int(bbox[0]), int(bbox[1] + 18))  # y roi 区域缩小15
        #         p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3] - 18))  # y roi 区域缩小10
        #
        #         # draw box
        #         text.rectangle(color_img, p1, p2, (255, 0, 0), 2, 1)
        #
        #         text.putText(color_img, "width:" + ("{:.1f}".format(W)), (p1[0], p1[1] + 20))
        #         text.putText(color_img, "height:" + ("{:.1f}".format(H)), (p1[0], p1[1] + 10))
        # print('f', frame)
        # frame = detecition(img)
        # blob = cv.dnn.blobFromImage(frame, swapRB=False, crop=False)
        # net.setInput(blob)
        # boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
        # print('b', boxes)
        # postprocess(boxes, masks)
        # t, _ = net.getPerfProfile()
        # label = 'Mask-RCNN Inference time for a frame : %0.0f ms' % abs(
        #     t * 1000.0 / cv.getTickFrequency())
        # cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        result = detect_mask.detect_with_h5(frame, depthFrame)
        cv.imshow("preview", frame)
        cv.imshow("result", result)
        key = cv.waitKey(1)
        if key == ord('q'):
            break
