import os
import cv2 as cv
import depthai as dai
from detect_mask import Mask
import time


# model_h5_path = "./logs/frozen_inference_graph_converted.pb"

width = 640
height = 400
confThreshold = 0.5  # 置信阈值
maskThreshold = 0.3  # Mask 阈值

# video setting

local_time = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
file_path = "./video/strawberry_" + local_time

folder = os.path.exists(file_path)
if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(file_path)  # makedirs 创建文件时如果路径不存在会创建这个路径
    print("---  new folder...  ---")
    print("---  OK  ---")
else:
    print("---  There is this folder!  ---")

excel_path = file_path + "/strawberry_result.xls"
video_path = file_path + "/strawberry_result.mp4"
rvideo_path = file_path + "/strawberry_rgb.mp4"
fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 指定视频视频编解码器格式
out = cv.VideoWriter(video_path, fourcc, 10, (width, height), True)
rout = cv.VideoWriter(rvideo_path, fourcc, 10, (width, height), True)


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
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    stero = pipeline.create(dai.node.StereoDepth)

    lrcheck = True
    subpixel = True

    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # 输出队列
    xoutPreview = pipeline.createXLinkOut()
    xoutDepth = pipeline.createXLinkOut()
    linkOut = pipeline.create(dai.node.XLinkOut)

    xoutPreview.setStreamName("preview")
    xoutDepth.setStreamName("depth")

    linkOut.setStreamName("sysinfo")

    # depth 节点设置
    stero.initialConfig.setConfidenceThreshold(255)
    stero.setDepthAlign(dai.CameraBoardSocket.RGB)
    stero.setLeftRightCheck(lrcheck)
    stero.setSubpixel(subpixel)


    # depth 后处理
    config = stero.initialConfig.get()
    config.postProcessing.speckleFilter.enable = False  # 是否启用或禁用过滤器。
    config.postProcessing.speckleFilter.speckleRange = 50  # 散斑搜索范围。
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 1
    config.postProcessing.thresholdFilter.minRange = 400
    config.postProcessing.thresholdFilter.maxRange = 15000
    config.postProcessing.decimationFilter.decimationFactor = 1
    stero.initialConfig.set(config)


    # 连接左右灰度x深度
    monoLeft.out.link(stero.left)
    monoRight.out.link(stero.right)

    # 空间计算
    stero.depth.link(xoutDepth.input)

    # bgr 输出
    camRgb.preview.link(xoutPreview.input)
    sysLog.setRate(1)
    sysLog.out.link(linkOut.input)

    return pipeline


pipeline = createPipeline()
with dai.Device(pipeline) as device:
    rgb = device.getOutputQueue('preview', maxSize=1, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)
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
        cv.imshow("rgb", frame)
        out.write(masked_image)
        rout.write(frame)


        key = cv.waitKey(1)
        if key == ord('q'):
            break
