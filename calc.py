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

    # def real_im(self, image, boxes, masks, class_ids, class_names,
    #                   scores=None, title="",
    #                   figsize=(16, 16), ax=None,
    #                   show_mask=True, show_bbox=True,
    #                   colors=None, captions=None):
    #
    #     N = boxes.shape[0]
    #
    #     if not N:
    #         print("\n*** No instances to display *** \n")
    #     else:
    #         assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    #
    #     # If no axis is passed, create one and automatically call show()
    #     auto_show = False
    #     if not ax:
    #         _, ax = plt.subplots(1, figsize=figsize)
    #         auto_show = True
    #
    #     # Generate random colors
    #     colors = colors or random_colors(N)
    #
    #     # Show area outside image boundaries.
    #     height, width = image.shape[:2]
    #     ax.set_ylim(height + 10, -10)
    #     ax.set_xlim(-10, width + 10)
    #     ax.axis('off')
    #     ax.set_title(title)
    #
    #     masked_image = image.astype(np.uint32).copy()
    #     for i in range(N):
    #         color = colors[i]
    #
    #         # Bounding box
    #         if not np.any(boxes[i]):
    #             # 跳过这个实例。没有bbox。可能在图像裁剪中丢失。
    #             continue
    #         y1, x1, y2, x2 = boxes[i]
    #         if show_bbox:
    #             p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
    #                                   alpha=0.7, linestyle="dashed",
    #                                   edgecolor=color, facecolor='none')
    #             ax.add_patch(p)
    #
    #         # area
    #         # 扁平化二维，却只行一元
    #         pre_masks = np.reshape(masks > .5, (-1, masks.shape[-1])).astype(np.float32)
    #         # 计算mask_面积
    #         area1 = np.sum(pre_masks, axis=0)
    #         # Label
    #         if not captions:
    #             class_id = class_ids[i]
    #             area = area1[i]
    #             score = scores[i] if scores is not None else None
    #             label = class_names[class_id]
    #             caption = "{} {:.3f}\narea:{:.1f}".format(label, score, area) if score else label
    #         else:
    #             caption = captions[i]
    #         ax.text(x1, y1 + 8, caption,
    #                 color='w', size=11, backgroundcolor="none")
    #
    #         # Mask
    #         mask = masks[:, :, i]
    #         if show_mask:
    #             masked_image = apply_mask(masked_image, mask, color)
    #
    #         # Mask Polygon
    #         # 垫以确保遮罩接触图像边缘的适当多边形。
    #         padded_mask = np.zeros(
    #             (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    #         padded_mask[1:-1, 1:-1] = mask
    #         contours = find_contours(padded_mask, 0.5)
    #         for verts in contours:
    #             # Subtract the padding and flip (y, x) to (x, y)
    #             verts = np.fliplr(verts) - 1
    #             p = Polygon(verts, facecolor="none", edgecolor=color)
    #             ax.add_patch(p)
    #     # print('mask', np.where(mask == 1))
    #     ax.imshow(masked_image.astype(np.uint8))
    #     if auto_show:
    #         plt.show()

    # ROI必须是整型数的列表
    def calc_spatials(self, depthFrame, roi, averaging_method=np.mean):
        roi = self._check_input(roi, depthFrame)  # 如果点通过，将其转换为ROI
        ymin, xmin, ymax, xmax = roi
        centerx = int((xmin + xmax)/2)
        centery = int((ymin + ymax)/2)
        cxmin = centerx - self.DELTA
        cxmax = centerx + self.DELTA
        cymin = centery - self.DELTA
        cymax = centery + self.DELTA
        # 计算ROI中的平均深度。
        # depthROI = depthFrame[int((3*ymin+ymax)/4):int((ymin+3*ymax)/4), int((3*xmin+xmax)/4):int((xmin+3*xmax)/4)]
        depthROI = depthFrame[ymin:ymax, xmin:xmax]
        inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)

        averageDepth = averaging_method(depthROI[inRange])

        centroid = {  # 获取ROI的质心
            'x': int((xmax + xmin) / 2),
            'y': int((ymax + ymin) / 2)
        }

        midW = int(depthFrame.shape[1] / 2)  # 中间深度img宽度
        midH = int(depthFrame.shape[0] / 2)  # 中间深度img高度
        bb_x_pos = centroid['x'] - midW
        bb_y_pos = centroid['y'] - midH
        Bw = int(math.fabs(xmin-xmax))
        Lh = int(math.fabs(ymin-ymax))

        angle_x = self._calc_angle(depthFrame, bb_x_pos)
        angle_y = self._calc_angle(depthFrame, bb_y_pos)
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
