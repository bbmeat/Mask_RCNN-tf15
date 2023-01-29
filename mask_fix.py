import math
import numpy as np
import matplotlib.pyplot as plt

class_names_ = ['left', 'right', 'bottom', 'top_left', 'top_right',
                'bottom_left', 'bottom_right', 'top']


def distance(point1, point2):
    x1 = point1[1]
    x2 = point2[1]
    y1 = point1[0]
    y2 = point2[0]
    dis = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dis


def midle(point1, point2):
    x1 = point1[1]
    x2 = point2[1]
    y1 = point1[0]
    y2 = point2[0]
    x = 0.5 * (x1 + x2)
    y = 0.5 * (y1 + y2)
    cent = [x, y]
    return cent


def arc(left_point, right_point, top):
    x1 = left_point[1]
    x2 = right_point[1]
    x3 = top[1]
    y1 = left_point[0]
    y2 = right_point[0]
    y3 = top[0]
    #  计算圆心坐标x0y0，以及半径r
    a = x1 - x2
    b = y1 - y2
    c = x1 - x3
    d = y1 - y3
    a1 = ((x1 * x1 - x2 * x2) + (y1 * y1 - y2 * y2)) / 2.0
    a2 = ((x1 * x1 - x3 * x3) + (y1 * y1 - y3 * y3)) / 2.0
    theta = b * c - a * d
    if abs(theta) < 1e-7:
        return -1
    x0 = (b * a2 - d * a1) / theta
    y0 = (c * a1 - a * a2) / theta
    r = np.sqrt(pow((x1 - x0), 2) + pow((y1 - y0), 2))

    # 计算圆心到扇形端点所在直线的距离
    line = [x1, y1, x2, y2]
    point = [x0, y0]
    line_point1, line_point2 = np.array(line[0:2]), np.array(line[2:])
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)

    ath = math.asin(r / distance)
    deg = math.degrees(ath)

    circle = (r ** 2) * math.pi
    area = ((180 - (deg * 2)) / 360) * circle
    return area


left = class_names_[0]
right = class_names_[1]
bottom = class_names_[2]
top_left = class_names_[3]
top_right = class_names_[4]
bottom_left = class_names_[5]
bottom_right = class_names_[6]
top = class_names_[7]

width = distance(left, right)
height = distance(top, bottom)
mid_width = midle(left, right)

top_width = distance(top_left, top_right)
mid_top = midle(top_left, top_right)
top_height = distance(top, mid_top)

bottom_width = distance(bottom_left, bottom_right)
mid_bottom = midle(bottom_left, bottom_right)
bottom_height = distance(bottom, mid_bottom)

up_height = distance(mid_top, mid_width)
down_height = distance(mid_width, mid_bottom)

up_mask_area = (math.fabs(width + top_width) * math.fabs(up_height) * 0.5)
down_mask_area = (math.fabs(width + bottom_width) * math.fabs(down_height) * 0.5)
top_area = arc(top_left, top_right, top)
bottom_area = arc(bottom_left, bottom_right, bottom)
area = up_mask_area + down_mask_area + top_area + bottom_area
x = []  # mask 面积
x = np.array(x)
y = []  # 近似模型预测函数
y = np.array(y)
f1 = np.polyfit(x, y, 4)  # 4次多项式拟合
print("f1:", f1)

p1 = np.poly1d(f1)  # 得到多项式系数，安阶数高低排序
print(p1)  # 显示多项式

yvalue = np.polyval(f1, x)

plt.plot(x, y, 'r*', label='original values')
plt.plot(x, yvalue, 'b+', label='polyfit values')
plt.xlabel('mask area')
plt.ylabel('real area')
plt.legend()
