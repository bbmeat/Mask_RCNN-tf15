import numpy as np


# a = np.array([[1, 0, 1, 0, 1, 1], [1, 0, 1, 0, 1, 0]])
def mask_count(array):
    a = array
    print(a > .5)
    masks = np.reshape(a > .5, (-1, 1)).astype(np.float32)  # 扁平化二维，却只行一元
    print('masks=', masks)
    area1 = np.sum(masks, axis=0)  # 计算mask_面积
    # print('mask_area1=', area1)
    # mask_intersections = np.dot(masks.T, masks)
    # print('mask_intersections=', mask_intersections)
    # union = area1[:, None] + area1[None, :] - mask_intersections
    # print(union)
    # iou_mask = union / mask_intersections
    #  print('iou_mask=', iou_mask)

    return area1
