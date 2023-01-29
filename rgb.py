#!/usr/bin/env python3

import cv2
import depthai as dai
import os
import time

width = 640
height = 400


# video setting
def time_now():
    time_lo = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    return time_lo


def save_img(dep_frame, num):
    local_time = time_now()
    file_name = "strawberry_" + local_time + "_" + str(num) + ".jpg"
    file_path = "./img_result/" + file_name
    print(file_name)
    cv2.imwrite(file_path, dep_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")

# Properties
camRgb.setPreviewSize(640, 400)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
camRgb.preview.link(xoutRgb.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    print('Connected cameras: ', device.getConnectedCameras())
    # Print out usb speed
    print('Usb speed: ', device.getUsbSpeed().name)

    # Output queue will be used to get the rgb frames from the output defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    frames = 0
    while True:
        inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived

        # Retrieve 'bgr' (opencv format) frame
        frame = inRgb.getCvFrame()
        cv2.imshow("rgb", inRgb.getCvFrame())
        frame_num = 1
        if frames % 100 == 0:
            frame_num += 1
            save_img(frame, frame_num)
        frames += 1
        if cv2.waitKey(1) == ord('q'):
            break
