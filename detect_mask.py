import math
import depthai as dai
import numpy as np

import strawberry_config
import cv2 as cv
import random
import colorsys
from utility import *
from skimage.measure import find_contours
import mrcnn.model as modellib

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
        self.config.display()
        self.model = modellib.MaskRCNN(mode="inference", config=self.config,
                                       model_dir='./logs')
        self.model.load_weights("G:/Python/Mask_RCNN-tf15/logs/mask_rcnn_shapes.h5", by_name=True)
        # threshold config
        self.detection_threshold = 0.5
        self.mask_threshold = 0.3

        self.class_name = []
        classesFile = "mscoco_labels.names";
        with open(classesFile, 'rt') as f:
            self.class_name = f.read().rstrip('\n').split('\n')

    def apply_mask(self, image, mask, color, alpha=0.5):
        """Apply the given mask to the image.
            """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c] * 255,
                                      image[:, :, c])
        return image

    def random_color(self, N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def _calc_angle(self, frame, offset):
        return math.atan(math.tan(self.monoHFOV / 2.0) * offset / (frame.shape[1] / 2.0))

    def detect_with_h5(self, bgr_frame, depth_frame):

        # config = InferenceConfig()

        image = cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB)
        result = self.model.detect([image])
        r = result[0]
        boxes = r['rois']
        masks = r['masks']
        class_ids = r['class_ids']
        class_names = self.class_name
        scores = r['scores']
        text = TextHelper

        N = boxes.shape[0]
        if not N:
            print("\n*** No instances to display *** \n")
        else:
            assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
        colors = self.random_color(N)
        masked_image = image.astype(np.uint32).copy()
        for i in range(N):
            color = colors[i]
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
            cv.rectangle(image, p1, p2, color, 1)

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
            depth_mm = depth_frame[centroid['y'], centroid['x']]
            Wd = int(math.fabs(x1 - x2))  # roi box width
            Hd = int(math.fabs(y1 - y2))  # roi box height
            Wb = self._calc_angle(depth_frame, Wd)
            Hb = self._calc_angle(depth_frame, Hd)
            roi_area = (depth_mm * math.tan(Wb)) * (depth_mm * math.tan(Hb))  # roi 区域的总体面积

            mask_area = (area / (Wd * Hd)) * roi_area

            spatials = {
                'z': depth_mm,
                'height': depth_mm * math.tan(Hb),
                'area': mask_area,
                'score': score,
                'name': label
            }

            # Mask

            mask = masks[:, :, i]
            masked_image = self.apply_mask(masked_image, mask, color)
            # Mask Polygon
            # 垫以确保遮罩接触图像边缘的适当多边形。
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
            masked_image = cv.fillConvexPoly(masked_image, verts, color=color)

            text.putText(masked_image, spatials['name'], (x1, y1 - 40))
            text.putText(masked_image, "Height: " + ("{:.1f}cm".format(spatials['height'])),
                         (x1, y1 - 25))
            text.putText(masked_image, "area: " + ("{:.1f}cm^2".format(spatials['area'] / 100)),
                         (x1 + 10, y1 - 30))
            text.putText(masked_image, "Z: " + ("{:.1f}cm".format(spatials['z'] / 10)),
                         (x1 + 10, y1 - 10))

        return masked_image
