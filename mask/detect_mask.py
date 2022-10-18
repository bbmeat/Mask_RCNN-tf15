import math
import depthai as dai
import numpy as np
import strawberry_config
import cv2 as cv
import random
import datetime
import colorsys
# from utility import *
from skimage.measure import find_contours
import mask.mrcnn.model as modellib

config = strawberry_config.StrawberryConfig()


class InferenceConfig(config.__class__):
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
        self.model.load_weights("G:/Python/Mask_RCNN-tf15/mask/logs/mask_rcnn_shapes.h5", by_name=True)
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
        classesFile = "mscoco_labels.names";
        with open(classesFile, 'rt') as f:
            self.class_name = f.read().rstrip('\n').split('\n')

    def apply_mask(self, image, mask, color, alpha=0):
        """Apply the given mask to the image.
            """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])
        return image

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

    def hvs_to_r(self, N, bright=True, alpha = 0.7):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv))
        colors = list(
            map(lambda x: (int(x[0] * 255 * alpha), int(x[1] * 255* alpha), int(x[2] * 255 * alpha)),
                colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        return colors

    def alpha_img(self, image, mask_image):
        alpha = 0.4  # first
        beta = 1 - alpha  # second
        gama = 0
        result = cv.addWeighted(image, alpha, mask_image, beta, gama)
        return result

    def _calc_angle(self, frame, offset):
        return math.atan(math.tan(self.monoHFOV / 2.0) * offset / (frame.shape[1] / 2.0))

    def _check_input(self, roi, frame):  # Check if input is ROI or point. If point, convert to ROI
        if len(roi) == 4: return roi
        if len(roi) != 2: raise ValueError("You have to pass either ROI (4 values) or point (2 values)!")
        # Limit the point so ROI won't be outside the frame
        self.DELTA = 5  # Take 10x10 depth pixels around point for depth averaging
        x = min(max(roi[0], self.DELTA), frame.shape[1] - self.DELTA)
        y = min(max(roi[1], self.DELTA), frame.shape[0] - self.DELTA)
        return (x - self.DELTA, y - self.DELTA, x + self.DELTA, y + self.DELTA)

    def calc_volume(self, mask, height, width):
        area = mask / 100
        h = height / 10
        w = width / 10
        vs = ((0.5 * w) ** 2) * h * math.pi
        # vs = format(v, '.1f')
        return vs

    def detect_with_h5(self, bgr_frame, depth_frame, averaging_method=np.mean):

        # config = InferenceConfig()

        # global mask_image

        image = cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB)
        a = datetime.datetime.now()
        result = self.model.detect([image])
        b = datetime.datetime.now()
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
            # print("\n*** No instances to display *** \n")
            return image
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
        print('n', N)
        colors = self.random_color(N)
        masked_image = image.astype(np.uint32).copy()
        bg_colors = self.hvs_to_r(N)
        for i in range(N):
            color = colors[i]
            bg_color = bg_colors[i]
            print('co', color)
            print('co2', bg_color)
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
            roi = [centroid['x'], centroid['y']]
            roi = self._check_input(roi, depth_frame)
            xmin, ymin, xmax, ymax = roi
            depthROI = depth_frame[ymin:ymax, xmin:xmax]
            inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)
            depth_mm = averaging_method(depthROI[inRange])

            Wd = int(math.fabs(x1 - x2))  # roi box width
            Hd = int(math.fabs(y1 - y2))  # roi box height
            Wb = self._calc_angle(depth_frame, Wd)
            Hb = self._calc_angle(depth_frame, Hd)
            roi_area = (depth_mm * math.tan(Wb)) * (depth_mm * math.tan(Hb))  # roi 区域的总体面积

            mask_area = (area / (Wd * Hd)) * roi_area

            spatials = {
                'z': depth_mm,
                'height': depth_mm * math.tan(Hb),
                'width': depth_mm * math.tan(Wb),
                'area': mask_area,
                'score': score,
                'name': label
            }

            # volume

            v = self.calc_volume(spatials['area'], spatials['height'], spatials['width'])

            # Mask

            mask = masks[:, :, i]
            # masked_image = self.apply_mask(masked_image, mask, color)
            _image = masked_image.astype(np.uint8)
            _yimg = image.astype(np.uint8)
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
                masked_image = cv.fillConvexPoly(_image, mask_aera, color=bg_color)
                # mask_image = self.alpha_img(_yimg, masked_image)
            # cv.imshow('img', _image)

            # x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
            cv.rectangle(masked_image, p1, p2, bg_color, 2)
            self.putText(masked_image, spatials['name'], (x1, y1 - 10))
            self.putText(masked_image, "area: " + ("{:.1f}cm^2".format(spatials['area'] / 100)), (x1, y1 - 70))
            self.putText(masked_image, "Height: " + ("{:.1f}cm".format(spatials['height'] / 10)), (x1, y1 - 55))
            self.putText(masked_image, "Z: " + ("{:.1f}cm".format(spatials['z'] / 10)), (x1, y1 - 40))
            self.putText(masked_image, "Vo: " + ("{:.1f}cm^3".format(v)), (x1, y1 - 25))
            self.putText(masked_image, "Inference time for a frame:{:.1f}ms".format(cost.microseconds / 1000), (0, 15))

        return masked_image
