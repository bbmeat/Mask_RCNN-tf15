import cv2 as cv
import depthai as dai
import time
import numpy as np

model_path = "./Model/frozen_inference_graph.pb";
model_pbTXT = "./Model/mask_rcnn.pbtxt"

# create pipeline
def createPipeline():
    print("creat pipeline")

    pipeline = dai.Pipeline()

    # rgb camera
    camRgb = pipeline.createColorCamera()
    camRgb.setPreviewSize(640, 400)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xoutPreview = pipeline.createXLinkOut()

    xoutPreview.setStreamName("preview")

    camRgb.preview.link(xoutPreview.input)

    return pipeline


def baseCamera(pipeline):
    with dai.Device() as device:
        device.startPipeline(pipeline)

        rgb = device.getOutputQueue('preview', maxSize=1, blocking=False)
        while True:
            preview = rgb.get()
            img = preview.getCvFrame()
            cv.imshow("preview", img)
            if cv.waitKey(1) == ord('q'):
                break


pipeline = createPipeline()
baseCamera(pipeline)
