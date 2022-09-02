import numpy as np


# a = np.array([[1, 0, 1, 0, 1, 1], [1, 0, 1, 0, 1, 0]])
def mask_count(masks):
    pre_masks = np.reshape(masks > .5, (-1, masks.shape[-1])).astype(np.float32)  # 扁平化二维，却只行一元

    area1 = np.sum(pre_masks, axis=0)  # 计算mask_面积

    return area1
