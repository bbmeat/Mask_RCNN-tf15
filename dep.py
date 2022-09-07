import numpy as np
import test_model
import math
import depthai as dai
import cv2

# StereoDepth initial config options.
outDepth = True  # Disparity by default
outConfidenceMap = True  # Output disparity confidence map
outRectified = True  # Output and display rectified streams
lrcheck = True  # Better handling for occlusions
extended = True  # Closer-in minimum depth, disparity range is doubled. Unsupported for now.
subpixel = False  # Better accuracy for longer distance, fractional disparity 32-levels

width = 640
height = 400

pipeline = dai.Pipeiline()

stereo = pipeline.create(dai.node.StereoDepth)
monoLeft = pipeline.create(dai.node.XLinkIn)
monoRight = pipeline.create(dai.node.XLinkIn)

xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutRectifLeft = pipeline.create(dai.node.XLinkOut)
xoutRectifRight = pipeline.create(dai.node.XLinkOut)
xoutStereoCfg = pipeline.create(dai.node.XLinkOut)

monoLeft.setStreamName('in_left')
monoRight.setStreamName('in_right')

xoutLeft.setStreamName('left')
xoutRight.setStreamName('right')
xoutDepth.setStreamName('depth')
xoutRectifLeft.setStreamName('rectified_left')
xoutRectifRight.setStreamName('rectified_right')
xoutStereoCfg.setStreamName('stereo_cfg')
# Properties
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
stereo.setLeftRightCheck(lrcheck)
stereo.setExtendedDisparity(extended)
stereo.setSubpixel(subpixel)
# allocates resources for worst case scenario
# allowing runtime switch of stereo modes
stereo.setRuntimeModeSwitch(True)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
stereo.syncedLeft.link(xoutLeft.input)
stereo.syncedRight.link(xoutRight.input)
stereo.depth.link(xoutDepth.input)
stereo.rectifiedLeft.link(xoutRectifLeft.input)
stereo.rectifiedRight.link(xoutRectifRight.input)

stereo.setInputResolution(width, height)
stereo.setRectification(False)
streams = ['left', 'right']

def calc(frame):
    baseline = 75
    HFOV = 71.8
    focal_length_in_pixels = width * 0.5 / math.tan(HFOV * 0.5 * math.PI / 180)

    depth = focal_length_in_pixels * baseline / frame
    return depth


print("Connecting and starting the pipeline")
# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    stereoDepthConfigInQueue = device.getInputQueue("steroDepthconfig")
    inStreams = ['in_right', 'in_left']
    inStreamsCameraID = [dai.CameraBoardSocket.RIGHT, dai.CameraBoardSocket.LEFT]
    in_q_list = []
    for s in inStreams:
        q = device.getInputQueue(s)
        in_q_list.append(q)

    # 为每个流创建一个接收队列
    q_list = []
    for s in streams:
        q = device.getOutputQueue(s, 8, blocking=False)
        q_list.append(q)

    inCfg = device.getOutputQueue("stereo_cfg", 8, blocking=False)