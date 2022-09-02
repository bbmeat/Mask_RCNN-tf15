import cv2
import depthai as dai
import numpy as np


def getFrame(queue):
    # Get frame from queue
    frame = queue.get()
    # Convert frame to OpenCV format and return
    return frame.getCvFrame()


# 选择黑白摄像头的函数
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


# 用于配置stereo pair的函数
def getStereoPair(pipeline, monoLeft, monoRight):
    # Configure stereo pair for depth estimation
    stereo = pipeline.createStereoDepth()

    # Checks occluded pixels and marks them as invalid
    stereo.setLeftRightCheck(True)

    # Configure left and right cameras to work as a stereo pair
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    return stereo


# 鼠标回调函数
def mouseCallback(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y


if __name__ == '__main__':
    mouseX = 0
    mouseY = 640

    # Start defining a pipeline
    pipeline = dai.Pipeline()

    # Set up left and right cameras
    monoLeft = getMonoCamera(pipeline, isLeft=True)
    monoRight = getMonoCamera(pipeline, isLeft=False)

    # Combine left and right cameras to form a stereo pair
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


with dai.Device(pipeline) as device:
    # 输出队列将用于从上面定义的输出中获取rgb帧和nn数据
    disparityQueue = device.getOutputQueue(name="disparity",
                                           maxSize=1, blocking=False)
    rectifiedLeftQueue = device.getOutputQueue(name="rectifiedLeft",
                                               maxSize=1, blocking=False)
    rectifiedRightQueue = device.getOutputQueue(name="rectifiedRight",
                                                maxSize=1, blocking=False)

    # 计算一个倍增器的颜色映射视差映射
    disparityMultiplier = 255 / stereo.initialConfig.getMaxDisparity()

    cv2.namedWindow("Stereo Pair")
    cv2.setMouseCallback("Stereo Pair", mouseCallback)

    # mapVariable用于在并排视图和一个帧视图之间切换。
    sideBySide = False
    while True:
        # Get the disparity map.
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
        imOut = cv2.line(imOut, (mouseX, mouseY),
                         (1280, mouseY), (0, 0, 255), 2)
        # Draw clicked point.
        imOut = cv2.circle(imOut, (mouseX, mouseY), 2,
                           (255, 255, 128), 2)
        cv2.imshow("Stereo Pair", imOut)
        cv2.imshow("Disparity", disparity)

        # Check for keyboard input
        key = cv2.waitKey(1)
        if key == ord('q'):
            # Quit when q is pressed
            break
        elif key == ord('t'):
            # Toggle display when t is pressed
            sideBySide = not sideBySide


