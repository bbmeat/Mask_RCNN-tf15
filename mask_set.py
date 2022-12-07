import colorsys
import math
import random
import time
from Ransac_Process import RANSAC
from PIL import Image
# from scipy.spatial import ConvexHull
import numpy as np
from strawberry_config import StrawberryConfig
import cv2 as cv
import matplotlib.pyplot as plt
import datetime
from rasnc import RANSAC as ran
# from utility import *
import skimage.io
import mrcnn.model as modellib
from skimage.draw import line
from skimage.measure import find_contours
import sympy
from sympy import *
from numpy.lib.type_check import iscomplex, real, imag, mintypecode


class InferenceConfig(StrawberryConfig):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False


# config_path = "./strawberry_config.py"

class Mask:
    # 校准相机数据
    def __init__(self):

        self.config = InferenceConfig()
        # self.config.display()
        self.model = modellib.MaskRCNN(mode="inference", config=self.config,
                                       model_dir='logs')
        self.model.load_weights("./logs/mask_rcnn_shapes.h5", by_name=True)
        # threshold config
        self.detection_threshold = 0.5
        self.mask_threshold = 0.3

        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv.FONT_HERSHEY_DUPLEX
        self.line_type = cv.LINE_AA
        self.class_name = []
        classes_file = "mscoco_labels.names"
        with open(classes_file, 'rt') as f:
            self.class_name = f.read().rstrip('\n').split('\n')

    def apply_mask(self, image, mask, alpha=1):
        """Apply the given mask to the image.
            """

        for c in range(3):
            image[:, :, c] = np.where(mask == 1, image[:, :, c] + 255, image[:, :, c])
        return image

    # def apply_mask(self, image, mask, color, alpha=0):
    #     """Apply the given mask to the image.
    #         """
    #     for c in range(3):
    #         image[:, :, c] = np.where(mask == 1, image[:, :, c] * 0 + color[c] * 255, image[:, :, c])
    #     return image

    def calc_volume(self, r, h):
        c = 2 * math.pi * r
        # volume = (c / 2) * (mask / 2)
        volume = ((r**2) * h) / 3
        print("体积：", volume)
        return volume

    def random_color(self, N, bright=True):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        # print('color', colors)
        return colors

    def myApprox(self, con):  # con为预先得到的最大轮廓
        num = 0.001
        # 初始化时不需要太小，因为四边形所需的值并不很小
        ep = num * cv.arcLength(con, True)
        con = cv.approxPolyDP(con, ep, True)
        # while (1):
        #     if len(con) <= 4:  # 防止程序崩溃设置的<=4
        #         break
        #     else:
        #         num = num * 1.5
        #         ep = num * cv.arcLength(con, True)
        #         con = cv.approxPolyDP(con, ep, True)
        #         continue
        return con

    def get_centor_line(self, image):
        global hull, approx1, volume
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
        v = []
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
            cont1_up = []
            cont13_up = []
            cont35_up = []
            cont1_down = []
            cont13_down = []
            cont35_down = []
            r = []
            for i in range(len(l_x)):
                conts = np.array(approx1[i, :][0])
                x = conts[0]
                y = conts[1]
                b = y - k2 * x
                x0 = int((b1 - b) / (k2 - k1))
                y0 = int(k2 * x0 + b)
                dis = self.distance(x, x0, y, y0)
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
            r_max = max(r)
            print("半径", r_max)

            print("cou1_up", cont1_up, "cou2_up", cont13_up, "cou3_up", cont35_up)
            print("cou1_down", cont1_down, "cou2_down", cont13_down, "cou3_down", cont35_down)
            cont1_up = np.array(cont1_up)
            cont13_up = np.array(cont13_up)
            cont35_up = np.array(cont35_up)
            cont1_down = np.array(cont1_down)
            cont13_down = np.array(cont13_down)
            cont35_down = np.array(cont35_down)
            if cont1_up.size != 0:
                top = self.mid_num(cont1_up)
            elif cont13_up.size != 0:
                top = self.mid_num(cont13_up)
            else:
                top = self.mid_num(cont35_up)
            print("top_point:", top)
            if cont1_down.size != 0:
                down = self.mid_num(cont1_down)
            elif cont13_down.size != 0:
                down = self.mid_num(cont13_down)
            else:
                down = self.mid_num(cont35_down)
            print("point_down", down)
            cv.circle(image, top, 1, (0, 0, 255), -1)
            cv.circle(image, down, 1, (0, 0, 255), -1)
            cv.line(image, top, down, (0, 255, 0), 2, cv.LINE_AA)
            # 果轴k
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
            height = self.distance(top[0], top[1], down[0], down[1])
            print("height:", height)
            volume = self.calc_volume(r_max, height)
            v.append(volume)

        # cv.drawContours(image, approx1, -1, (255, 0, 0), 1)

        cv.imshow("img", image)
        cv.waitKey(0)
        return image, v

    # def y_to_x(self, k, b):
    #     # y = symbols('y')
    #     # x = f[0] + f[1] * y + f[2] * (y ** 2) + f[3] * (y ** 3)
    #     x = symbols('x')
    #     y = k * x + b
    #     # x = y / k - b / k
    #     k2 = 1 / k
    #     b2 = -1 * (b / k)
    #
    #     return k2, b2
    #
    # def cala_deg_volume(self, fx, k, b, d_top, d_down):
    #     A = k
    #     C = b
    #     f = []
    #     for i in fx:
    #         k_x = sympy.sympify(i)
    #         f.append(k_x)
    #     # d_top = point_list[0]
    #     # d_top = min(point_list)
    #     # # d_down = point_list[-1]
    #     # d_down = max(point_list)
    #     x = symbols('x')
    #     y = f[0] + f[1] * x + f[2] * (x ** 2)
    #     # y = f[0] + f[1] * x + f[2] * (x ** 2) + f[3] * (x ** 3)
    #     fx_d = diff(y, x)
    #     fx_n = ((y - A * x - C) ** 2) * abs(A * fx_d + 1)
    #     fx_djf = integrate(fx_n, (x, d_down, d_top))
    #     # print(fx_djf)
    #     volume = (math.pi / ((math.sqrt(A + 1)) ** 3)) * fx_djf
    #     print("体积：", volume)
    #     return volume

    def distance(self, x, x0, y, y0):
        dis = math.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        return dis

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

    def detect_with_h5(self, bgr_frame):

        # image = cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB)
        global masked_image, mask_aera, area1
        result = self.model.detect([bgr_frame])
        r = result[0]
        boxes = r['rois']
        masks = r['masks']
        class_ids = r['class_ids']
        # text = TextHelper

        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
            return bgr_frame
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
        print('n', N)
        colors = self.random_color(N)
        image = bgr_frame.astype(np.uint32).copy()
        mask_image = image[:, :, 0] + image[:, :, 1] + image[:, :, 2]
        image[mask_image != 0] = 0
        for i in range(N):
            color = colors[i]
            if not np.any(boxes[i]):
                # 跳过这个实例。没有bbox。可能在图像裁剪中丢失了。
                continue

            # area
            pre_masks = np.reshape(masks > .5, (-1, masks.shape[-1])).astype(np.float32)
            # 计算mask_面积
            area1 = np.sum(pre_masks, axis=0)
            print("area", area1)
            mask = masks[:, :, i]
            masked_image = self.apply_mask(image, mask)
            # masked_image = self.apply_mask(image, mask, color)
            masked_image = masked_image.astype(np.uint8)

        imag, volume = self.get_centor_line(masked_image)
        print("vo", volume)

        return masked_image


def main():
    # image_path = skimage.io.imread("./train_data/val/000000000001.png")
    # image_path = skimage.io.imread("./train_data/val/000000000002.png")
    # image_path = skimage.io.imread("./train_data/val/000000000008.png")
    # image_path = skimage.io.imread("./train_data/val/000000000009.png")
    # image_path = skimage.io.imread("./train_data/val/rgb2.png")
    image_path = skimage.io.imread("./train_data/val/rgb3.png")
    image = Mask().detect_with_h5(image_path)
    # image = image.astype(np.uint8)
    # cv.imshow("result", image)
    # cv.waitKey(0)


if __name__ == "__main__":
    main()
