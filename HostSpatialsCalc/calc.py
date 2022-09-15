import math
import numpy as np
import depthai as dai
import test_model

class HostSpatialsCalc:
    # 我们需要设备对象来获取校准数据
    def __init__(self, device):
        calibData = device.readCalibration()
        # 计算主机空间坐标所需的信息
        self.monoHFOV = np.deg2rad(calibData.getFov(dai.CameraBoardSocket.LEFT))
        # Values
        self.DELTA = 5
        self.THRESH_LOW = 200  # 20cm
        self.THRESH_HIGH = 30000  # 30m

    def setLowerThreshold(self, threshold_low):
        self.THRESH_LOW = threshold_low

    def setUpperThreshold(self, threshold_low):
        self.THRESH_HIGH = threshold_low

    def _check_input(self, roi, frame):  # 检查输入是ROI还是point。如果点，转换为ROI
        if len(roi) == 4: return roi
        if len(roi) != 2: raise ValueError("You have to pass either ROI (4 values) or point (2 values)!")
        # 限制点，所以ROI不会在框架之外
        self.DELTA = 5  # 在点周围取10x10深度像素进行深度平均
        x = min(max(roi[0], self.DELTA), frame.shape[1] - self.DELTA)
        y = min(max(roi[1], self.DELTA), frame.shape[0] - self.DELTA)
        return (x - self.DELTA, y - self.DELTA, x + self.DELTA, y + self.DELTA)

    def _calc_angle(self, frame, offset):
        return math.atan(math.tan(self.monoHFOV / 2.0) * offset / (frame.shape[1] / 2.0))

    # ROI必须是整型数的列表
    def calc_spatials(self, depthFrame, roi, averaging_method=np.mean):
        roi = self._check_input(roi, depthFrame)  # 如果点通过，将其转换为ROI
        xmin, ymin, xmax, ymax = roi
        # ymin, xmin, ymax, xmax = roi
        centerx = int((xmin + xmax)/2)
        centery = int((ymin + ymax)/2)
        self.DELTA = 5  # 在点周围取10x10深度像素进行深度平均
        x = min(max(centerx, self.DELTA), depthFrame.shape[1] - self.DELTA)
        y = min(max(centery, self.DELTA), depthFrame.shape[0] - self.DELTA)
        cxmin = x - self.DELTA
        cxmax = x + self.DELTA
        cymin = y - self.DELTA
        cymax = y + self.DELTA
        # 计算ROI中的平均深度。
        # depthROI = depthFrame[int((3*ymin+ymax)/4):int((ymin+3*ymax)/4), int((3*xmin+xmax)/4):int((xmin+3*xmax)/4)]
        depthROI = depthFrame[cymin:cymax, cxmin:cxmax]
        inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)

        averageDepth = averaging_method(depthROI[inRange]) - 12

        centroid = {  # 获取ROI的质心
            'x': int((xmax + xmin) / 2),
            'y': int((ymax + ymin) / 2)
        }

        Bw = int(math.fabs(xmin-xmax))
        Lh = int(math.fabs(ymin-ymax))

        Hb = self._calc_angle(depthFrame, Lh)
        Wb = self._calc_angle(depthFrame, Bw)
        area = (averageDepth * math.tan(Hb)) * (averageDepth*math.tan(Wb))
        mask_area = (test_model.area1/(Bw * Lh)) * area
        spatials = {
            'z': averageDepth,
            'h': averageDepth * math.tan(Hb),
            'a': mask_area
        }

        # print('zb', spatials['z'])
        return spatials, centroid
