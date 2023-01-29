import os
import time
import schedule
import math
import depthai as dai
import numpy as np
from strawberry_config import StrawberryConfig
import cv2 as cv
import random
import datetime
import colorsys
import json
import requests
# from utility import *
from skimage.measure import find_contours
import mrcnn.model as modellib


class InferenceConfig(StrawberryConfig):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False


# config_path = "./strawberry_config.py"


class Mask:
    # 校准相机数据
    def __init__(self, device):
        calibData = device.readCalibration()
        # 计算主机空间坐标所需的信息
        self.monoHFOV = np.deg2rad(calibData.getFov(dai.CameraBoardSocket.LEFT))
        self.config = InferenceConfig()
        # self.config.display()
        self.model = modellib.MaskRCNN(mode="inference", config=self.config,
                                       model_dir='logs')
        self.model.load_weights("./logs/mask_rcnn_strawberry.h5", by_name=True)
        # threshold config
        self.detection_threshold = 0.5
        self.mask_threshold = 0.3
        self.THRESH_LOW = 200  # 20cm
        self.THRESH_HIGH = 30000  # 30m
        # text config
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv.FONT_HERSHEY_DUPLEX
        self.line_type = cv.LINE_AA
        self.class_name = []
        classes_file = "mscoco_labels.names";
        with open(classes_file, 'rt') as f:
            self.class_name = f.read().rstrip('\n').split('\n')

    def apply_mask(self, image, mask, color, alpha=0):
        """Apply the given mask to the image.
            """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])
        return image

    def random_color(self, N, bright=True):

        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def putText(self, frame, text, coords):
        cv.putText(frame, text, coords, self.text_type, 0.5, self.bg_color, 3, self.line_type)
        cv.putText(frame, text, coords, self.text_type, 0.5, self.color, 1, self.line_type)

    def back_color(self):

        R = random.randrange(50, 256)
        G = random.randrange(50, 256)
        B = random.randrange(50, 256)
        colors = (R, G, B)
        return colors

    def hvs_to_r(self, N, bright=True, alpha=0.7):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv))
        colors = list(
            map(lambda x: (int(x[0] * 255 * alpha), int(x[1] * 255 * alpha), int(x[2] * 255 * alpha)),
                colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        return colors

    def _calc_angle(self, frame, offset):
        result = math.atan((math.tan(self.monoHFOV / 2.0) * offset) / (frame.shape[1] / 2.0))
        return result

    def _check_input(self, roi, frame):  # Check if input is ROI or point. If point, convert to ROI
        if len(roi) == 4: return roi
        if len(roi) != 2: raise ValueError("You have to pass either ROI (4 values) or point (2 values)!")
        # Limit the point so ROI won't be outside the frame
        self.DELTA = 5  # Take 10x10 depth pixels around point for depth averaging
        x = min(max(roi[0], self.DELTA), frame.shape[1] - self.DELTA)
        y = min(max(roi[1], self.DELTA), frame.shape[0] - self.DELTA)
        return (x - self.DELTA, y - self.DELTA, x + self.DELTA, y + self.DELTA)

    def calc_volume(self, r, h):
        # c = 2 * math.pi * r
        # volume = (c / 2) * (mask / 2)

        v = ((r ** 2) * h * math.pi) / 3
        volume = 0.01764 * (v ** 2) + 0.6937 * v + 4.314
        print("体积：", volume)
        return volume

    def calc_dep_all(self, centroid, depth_frame, averaging_method=np.mean):
        roi = [centroid['x'], centroid['y']]
        roi = self._check_input(roi, depth_frame)
        xmin, ymin, xmax, ymax = roi
        depthROI = depth_frame[ymin:ymax, xmin:xmax]
        inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)

        depth_mm = averaging_method(depthROI[inRange])
        print("dep", depth_mm / 10)
        return depth_mm

    def calc_dep(self, centroid, depth_frame, area, score, label, boxes):
        y1, x1, y2, x2 = boxes
        depth_mm = self.calc_dep_all(centroid, depth_frame)
        if str(depth_mm) == 'nan':
            print("depth not enough")
            spatials = {
                'z': -1,
                'area': -1,
                'score': -1,
                'name': label
            }
            return spatials
        else:
            depth_mm = depth_mm
        Wd = int(math.fabs(x1 - x2))  # roi box width
        Hd = int(math.fabs(y1 - y2))  # roi box height
        Wb = self._calc_angle(depth_frame, Wd)
        Hb = self._calc_angle(depth_frame, Hd)
        roi_area = (depth_mm * math.tan(Wb)) * (depth_mm * math.tan(Hb))  # roi 区域的总体面积

        mask_area = ((area / (Wd * Hd)) * roi_area) / 100

        spatials = {
            'z': depth_mm / 10,
            # 'height': (depth_mm * math.tan(height)) / 10,
            # 'width': depth_mm * math.tan(Wb) / 10,
            'area': mask_area,
            'score': score,
            'name': label
        }

        # v = self.calc_volume(spatials['height'], spatials['width'])
        # spatials['volume'] = volume

        print(spatials)
        return spatials

    def distance(self, x, x0, y, y0, depth_frame, depth_mm):
        # depth_mm = self.calc_dep_all(centroid, depth_frame)
        distan = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        if distan == 0:
            dist = int(math.fabs(1e-10))
        else:
            dist = int(math.fabs(distan))
        dist_d = self._calc_angle(depth_frame, dist)
        dis = depth_mm * math.tan(dist_d) / 10
        return dis

    def apply_mask_bw(self, image, mask, alpha=1):
        """Apply the given mask to the image.
            """

        for c in range(3):
            image[:, :, c] = np.where(mask == 1, image[:, :, c] + 255, image[:, :, c])
        return image

    def mid_num(self, ary):
        ary_list = np.array(ary)
        x = []
        y = []
        for i in range(len(ary_list)):
            x.append(ary[i][0])
            y.append(ary[i][1])
        x = int(np.mean(np.array(x)))
        y = int(np.mean(np.array(y)))
        point = (x, y)
        return point

    def get_centor_line(self, image, depth, centroid):
        global hull, height
        volume = []
        height = []
        image = np.uint8(image)
        # print(image)
        img = cv.cvtColor(np.uint8(image), cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(img, 230, 255, cv.THRESH_BINARY_INV)
        # OpenCV定义的结构矩形元素
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        # 膨胀图像
        dilated = cv.dilate(thresh, kernel)
        # 闭运算
        # closed3 = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel, iterations=3)
        contours, hierarchy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # print("hierarchy", hierarchy)
        # print("contours", contours)

        for c in contours:
            if len(c) < 10:  # 去除外轮廓
                continue
            # 找到边界坐标
            # print("c", c)

            hull = cv.convexHull(c)
            # print("hull", hull)
            # 逼近多边形
            # epsilon = 0.001 * cv.arcLength(c, True)
            approx1 = c
            # approx1 = self.myApprox(c)  # 拟合精确度
            # print("app", approx1)

            # print("cv2.approxPolyDP()：", cv.isContourConvex(approx1))
            # img1 = cv.polylines(image, [approx1], True, (255, 255, 0), 2)
            x, y, w, h = cv.boundingRect(approx1)  # 计算点集最外面的矩形边界
            l_x = np.array(approx1)
            # cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 绿
            p1 = (x, y)
            p2 = (x + w, y + h)
            p3 = (x + w, y)
            p4 = (x, y + h)
            print("rect_point", [p1, p2, p3, p4])
            # 计算每个轮廓的矩
            M = cv.moments(approx1)

            # 计算质心的x y坐标
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print("barycenter", (cX, cY))
            # centroid = {  # 获取ROI的质心
            #     'x': cX,
            #     'y': cY
            # }
            depth_mm = self.calc_dep_all(centroid, depth)
            if str(depth_mm) == 'nan':
                print("depth not enough")
                height = -1
                volume = -1
                return height, volume
            else:
                depth_mm = depth_mm
            cv.circle(image, (cX, cY), 3, (0, 255, 0), -1)
            ellipse = cv.fitEllipse(approx1)
            print("ty", ellipse)
            # opencv 拟合椭圆
            cen_x, cen_y = ellipse[0]
            # cv.ellipse(image, ellipse, (255, 0, 0), 2)
            cv.circle(image, (int(cen_x), int(cen_y)), 2, (255, 0, 0), -1)

            # 角度计算斜率
            # 角度转弧度
            angle = ellipse[2] - 90
            # angle = Angle - 90
            # angle_radian = math.radians(angle)
            angle_radian = angle / 180 * math.pi

            print("角度", angle)
            print("弧度", angle_radian)
            # print("角度", orientation_rads)
            # 斜率
            k1 = math.tan(angle_radian)
            print("方向斜率", k1)
            # 直线
            b1 = cY - k1 * cX
            # 垂直果轴的线
            cont1_up = []
            cont13_up = []
            cont35_up = []
            cont5u_up = []
            cont1_down = []
            cont13_down = []
            cont35_down = []
            cont5u_down = []
            r = []
            r_real = []
            if k1 == 0:
                pp3 = (cX, cY - 10)
                pp4 = (cX, cY + 10)
                cv.line(image, pp3, pp4, (0, 255, 0), 2, cv.LINE_AA)
                pa, pb = (x, int(k1 * x + b1)), ((x + w), int(k1 * (x + w) + b1))
                print("pa", pa, "pb", pb)
                for i in range(len(l_x)):
                    conts = np.array(approx1[i, :][0])
                    x = conts[0]
                    y = conts[1]
                    y0 = y
                    dis = math.fabs(cY - y0)
                    r_r = self.distance(cX, x, cY, y0, depth, depth_mm)
                    r_real.append(r_r)
                    r.append(dis)
                    if y0 > cY:  # 设定点到直线的距离范围。
                        # dis = self.distance(x, x0, y, y0)
                        # print(">distance", dis)
                        if dis <= 1:
                            cont1_up.append(conts)
                        elif 1 < dis <= 3:
                            cont13_up.append(conts)
                        elif 3 < dis <= 5:
                            cont35_up.append(conts)
                    else:
                        # dis = self.distance(x, x0, y, y0)
                        # print("<distance", dis)
                        if dis <= 1:
                            cont1_down.append(conts)
                        elif 1 < dis <= 3:
                            cont13_down.append(conts)
                        elif 3 < dis <= 5:
                            cont35_down.append(conts)
                # r_max = max(r_real)
                # print("半径", r_max)
            else:
                if b1 == 0:
                    k2 = 0
                    b2 = cY - k2 * cX
                    print("xl", k1)
                    pp1 = ((cX - 10), int(k1 * (cX - 10) + b1))
                    pp2 = ((cX + 10), int(k1 * (cX + 10) + b1))
                    cv.line(image, pp1, pp2, (255, 0, 0), 2, cv.LINE_AA)

                    pp3 = ((cX - 10), int(k2 * (cX - 10) + b2))
                    pp4 = ((cX + 10), int(k2 * (cX + 10) + b2))
                    cv.line(image, pp3, pp4, (0, 255, 0), 2, cv.LINE_AA)
                    pa, pb = (x, int(k1 * x + b1)), ((x + w), int(k1 * (x + w) + b1))
                    print("pa", pa, "pb", pb)
                    # 计算上焦点和下焦点

                    for i in range(len(l_x)):
                        conts = np.array(approx1[i, :][0])
                        x = conts[0]
                        y = conts[1]
                        y0 = y
                        dis = math.fabs(cX - x)
                        r_r = self.distance(x, cX, y, cY, depth, depth_mm)
                        r.append(dis)
                        r_real.append(r_r)
                        if x > cX:  # 设定点到直线的距离范围。
                            # dis = self.distance(x, x0, y, y0)
                            # print(">distance", dis)
                            if dis <= 1:
                                cont1_up.append(conts)
                            elif 1 < dis <= 3:
                                cont13_up.append(conts)
                            elif 3 < dis <= 5:
                                cont35_up.append(conts)
                            else:
                                cont5u_up.append(conts)
                        else:
                            # dis = self.distance(x, x0, y, y0)
                            # print("<distance", dis)
                            if dis <= 1:
                                cont1_down.append(conts)
                            elif 1 < dis <= 3:
                                cont13_down.append(conts)
                            elif 3 < dis <= 5:
                                cont35_down.append(conts)
                            else:
                                cont5u_down.append(conts)
                        # r_max = max(r_real)
                        # print("半径", r_max)
                else:
                    k2 = -1 / k1
                    b2 = cY - k2 * cX
                    print("xl", k1)
                    pp1 = ((cX - 10), int(k1 * (cX - 10) + b1))
                    pp2 = ((cX + 10), int(k1 * (cX + 10) + b1))
                    cv.line(image, pp1, pp2, (255, 0, 0), 2, cv.LINE_AA)

                    pp3 = ((cX - 10), int(k2 * (cX - 10) + b2))
                    pp4 = ((cX + 10), int(k2 * (cX + 10) + b2))
                    cv.line(image, pp3, pp4, (0, 255, 0), 2, cv.LINE_AA)
                    pa, pb = (x, int(k1 * x + b1)), ((x + w), int(k1 * (x + w) + b1))
                    print("pa", pa, "pb", pb)
                    # 计算上焦点和下焦点

                    for i in range(len(l_x)):
                        conts = np.array(approx1[i, :][0])
                        x = conts[0]
                        y = conts[1]
                        b = y - k2 * x
                        x0 = int((b1 - b) / (k2 - k1))
                        y0 = int(k2 * x0 + b)
                        r_r = self.distance(x, x0, y, y0, depth, depth_mm)
                        distan = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                        dis = math.fabs(int(distan))
                        r.append(dis)
                        r_real.append(r_r)
                        if y0 > cY:  # 设定点到直线的距离范围。
                            # dis = self.distance(x, x0, y, y0)
                            # print(">distance", dis)
                            if dis <= 1:
                                cont1_up.append(conts)
                            elif 1 < dis <= 3:
                                cont13_up.append(conts)
                            elif 3 < dis <= 5:
                                cont35_up.append(conts)
                            else:
                                cont5u_up.append(conts)
                        else:
                            # dis = self.distance(x, x0, y, y0)
                            # print("<distance", dis)
                            if dis <= 1:
                                cont1_down.append(conts)
                            elif 1 < dis <= 3:
                                cont13_down.append(conts)
                            elif 3 < dis <= 5:
                                cont35_down.append(conts)
                            else:
                                cont5u_down.append(conts)
            r_max = max(r_real)
            print("半径", r_max)

            print("cou1_up", cont1_up, "cou2_up", cont13_up, "cou3_up", cont35_up, "cont5u_up", cont5u_up)
            print("cou1_down", cont1_down, "cou2_down", cont13_down, "cou3_down", cont35_down, "cont5u_down",
                      cont5u_down)
            cont1_up = np.array(cont1_up)
            cont13_up = np.array(cont13_up)
            cont35_up = np.array(cont35_up)
            cont5u_up = np.array(cont5u_up)
            cont1_down = np.array(cont1_down)
            cont13_down = np.array(cont13_down)
            cont35_down = np.array(cont35_down)
            cont5u_down = np.array(cont5u_down)
            if cont1_up.size != 0:
                top = self.mid_num(cont1_up)
            elif cont13_up.size != 0:
                top = self.mid_num(cont13_up)
            elif cont35_up.size != 0:
                top = self.mid_num(cont35_up)
            elif cont5u_up.size != 0:
                top = self.mid_num(cont5u_up)
            else:
                top = pa
            print("top_point:", top)
            if cont1_down.size != 0:
                down = self.mid_num(cont1_down)
            elif cont13_down.size != 0:
                down = self.mid_num(cont13_down)
            elif cont35_down.size != 0:
                down = self.mid_num(cont35_down)
            elif cont5u_down.size != 0:
                down = self.mid_num(cont5u_down)
            else:
                down = pb
            print("point_down", down)
            cv.circle(image, top, 1, (0, 0, 255), -1)
            cv.circle(image, down, 1, (0, 0, 255), -1)
            cv.line(image, top, down, (0, 255, 0), 2, cv.LINE_AA)
            # 果轴k
            if (down[0] - top[0]) == 0:
                k_n = 0
            else:
                k_n = (down[1] - top[1]) / (down[0] - top[0])
            b_n = top[1] - k_n * top[0]
            print("斜率：", k_n)

            # 对角斜率
            k_p = (p4[1] - cY) / (p4[0] - cX)
            print("对角斜率：", k_p)
            # 果轴角度
            ang_g = math.degrees(math.atan(k_n))
            ang_d = math.degrees(math.atan(k_p))
            print("对角角度", ang_d, "果轴角度", ang_g)

            if ang_g > 0:
                ang_d_abs = abs(ang_d)
                ang_g_abs = abs(ang_g)
            else:
                ang_d_abs = 90 - abs(ang_d)
                ang_g_abs = 90 - abs(ang_g)
            print("对角绝对角度", ang_d_abs, "果轴绝对角度", ang_g_abs)
            # 计算高度
            h = self.distance(top[0], down[0], top[1], down[1], depth, depth_mm)
            h1 = h * 0.9366 + 1.334
            height.append(h1)
            print("height:", h1)
            v = self.calc_volume(r_max, h1)
            volume.append(v)

        # cv.drawContours(image, approx1, -1, (255, 0, 0), 1)
        # volume = volume.reverse()
        # height = height.reverse()
        # cv.imshow("img", image)
        # cv.waitKey(0)
        return height, volume, (top, down)

    def detect_with_h5(self, bgr_frame, depth_frame, color_dep_frame):

        # config = InferenceConfig()

        # global mask_image

        global strawberry_data, label, masked_image_bw, cen_point
        image = cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB)
        a = datetime.datetime.now()
        result = self.model.detect([image])
        b = datetime.datetime.now()
        strawberry_data = []
        cost = b - a
        # fps = 1.0 / cost.microseconds
        r = result[0]
        boxes = r['rois']
        masks = r['masks']
        class_ids = r['class_ids']
        class_names = self.class_name
        scores = r['scores']
        # text = TextHelper

        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
            return (image, color_dep_frame), strawberry_data
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
        print('n', N)
        # colors = self.random_color(N)
        masked_image = image.astype(np.uint32).copy()
        bg_colors = self.hvs_to_r(N)

        for i in range(N):
            # color = colors[i]
            bg_color = bg_colors[i]
            # print('co', color)
            # print('co2', bg_color)
            if not np.any(boxes[i]):
                # 跳过这个实例。没有bbox。可能在图像裁剪中丢失了。
                continue
            y1, x1, y2, x2 = boxes[i]

            centroid = {  # 获取ROI的质心
                'x': int((x1 + x2) / 2),
                'y': int((y1 + y2) / 2)
            }

            print('box', boxes)
            p1 = (int(x1), int(y1))
            p2 = (int(x2), int(y2))

            # area
            pre_masks = np.reshape(masks > .5, (-1, masks.shape[-1])).astype(np.float32)
            # 计算mask_面积
            area1 = np.sum(pre_masks, axis=0)

            # label
            class_id = class_ids[i]
            area = area1[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]

            # real data
            # Mask
            image_bw = bgr_frame.astype(np.uint32).copy()
            mask_image = image_bw[:, :, 0] + image_bw[:, :, 1] + image_bw[:, :, 2]
            image_bw[mask_image != 0] = 0
            mask = masks[:, :, i]
            masked_image_bw = self.apply_mask_bw(image_bw, mask)
            masked_image = masked_image.astype(np.uint8)

            spatials = self.calc_dep(centroid, depth_frame, area, score,
                                     label, boxes[i])

            if spatials['z'] == -1:
                strawberry_name = label + str(i + 1)
                st_data = {
                    "class_name": label,
                    "id_name": strawberry_name,
                    "height": -1,
                    "area": -1,
                    "volume": -1,
                    "weight": -1
                }
            else:
                heigh, vol, cen_point = self.get_centor_line(masked_image_bw,
                                                  depth_frame, centroid)
                weight = "{:.1f}".format(vol[0] * 0.99)
                # height = "{:.1f}".format(spatials['height'])
                area = "{:.1f}".format(spatials['area'])
                # vo = "{:.1f}".format(spatials['volume'])
                spatials['height'] = heigh[0]
                spatials['volume'] = vol[0]
                strawberry_name = label + str(i + 1)
                st_data = {
                    "class_name": label,
                    "id_name": strawberry_name,
                    "area": area,
                    "height": heigh[0],
                    "volume": vol[0],
                    "weight": weight
                }
            strawberry_data.append(st_data)

            # print("data", strawberry_data)

            # Mask Polygon
            # 垫以确保遮罩接触图像边缘的适当多边形。
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            # print('p', contours)

            for verts in contours:
                # 减去填充并将(y, x)翻转到(x, y)
                verts = np.fliplr(verts) - 1
                # p = Polygon(verts, facecolor="none", edgecolor=color)

                mask_aera = verts.astype(int)
                masked_image = cv.fillConvexPoly(masked_image, mask_aera, color=bg_color)

            cv.rectangle(masked_image, p1, p2, bg_color, 4)
            cv.rectangle(color_dep_frame, p1, p2, bg_color, 4)
            self.putText(masked_image, spatials['name'], (x1, y1 - 10))
            cv.line(masked_image, cen_point[0], cen_point[1], (0, 255, 0), 2, cv.LINE_AA)
            # self.putText(masked_image, "Area: " + ("{:.1f}cm^2".format(spatials['area'])), (x1, y1 - 70))
            # self.putText(masked_image, "Height: " + ("{:.1f}cm".format(spatials['height'])), (x1, y1 - 40))
            self.putText(masked_image, "Height: " + ("{:.1f}cm".format(spatials['height'])), (x1, y1 - 40))
            # self.putText(masked_image, "Z: " + ("{:.1f}cm".format(spatials['z'] / 10)), (x1, y1 - 40))
            # self.putText(masked_image, "Volume: " + ("{:.1f}cm^3".format(spatials['volume'])), (x1, y1 - 25))
            self.putText(masked_image, "Volume: " + ("{:.1f}cm^3".format(spatials['volume'])), (x1, y1 - 25))
            self.putText(masked_image, "Inference time for a frame:{:.1f}ms".format(cost.microseconds / 1000), (0, 15))

        return (masked_image, color_dep_frame), strawberry_data
