import numpy as np
import cv2


def Giou_np(prd_mask, gt_mask):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    pos_prd_mask = np.where(prd_mask == 1)
    pos_gt_mask = np.where(gt_mask == 1)

    bbox_p = [np.min(pos_prd_mask[0]),
              np.min(pos_prd_mask[1]),
              np.max(pos_prd_mask[0]),
              np.max(pos_prd_mask[1])]

    bbox_g = [np.min(pos_gt_mask[0]),
              np.min(pos_gt_mask[1]),
              np.max(pos_gt_mask[0]),
              np.max(pos_gt_mask[1])]

    # calc area of Bg
    area_p = (bbox_p[2] - bbox_p[0]) * (bbox_p[3] - bbox_p[1])
    # calc area of Bp
    area_g = (bbox_g[2] - bbox_g[0]) * (bbox_g[3] - bbox_g[1])

    # cal intersection
    x1I = np.maximum(bbox_p[0], bbox_g[0])
    y1I = np.maximum(bbox_p[1], bbox_g[1])
    x2I = np.minimum(bbox_p[2], bbox_g[2])
    y2I = np.minimum(bbox_p[3], bbox_g[3])
    I = np.max((y2I - y1I), 0) * np.max((x2I - x1I), 0)

    # find enclosing box
    x1C = np.minimum(bbox_p[0], bbox_g[0])
    y1C = np.minimum(bbox_p[1], bbox_g[1])
    x2C = np.maximum(bbox_p[2], bbox_g[2])
    y2C = np.maximum(bbox_p[3], bbox_g[3])

    # calc area of Bc
    area_c = (x2C - x1C) * (y2C - y1C)
    U = area_p + area_g - I
    #iou = 1.0 * I / U

    # Giou


    gt_mask = gt_mask.astype('float64')
    img_and = cv2.bitwise_and(prd_mask, gt_mask)
    img_or = cv2.bitwise_or(prd_mask, gt_mask)
    j = np.count_nonzero(img_and)
    k = np.count_nonzero(prd_mask)
    p_mask = float(float(j)/(float(k)+1))

    giou = p_mask - (area_c - U) / area_c

    return giou, p_mask



'''
def calculate_iou(img_mask, gt_mask):
    gt_mask = gt_mask.astype('float64')
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    img_or = cv2.bitwise_or(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(img_or)
    k = np.count_nonzero(img_mask)
    iou = float(float(j)/(float(i)+1))
    # power of mask
    p_mask = float(float(j)/(float(k)+1))
    return iou, p_mask
'''

def calculate_overlapping(img_mask, gt_mask):
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(gt_mask)
    if i>0:
        overlap = float(float(j)/float(i))
    else:
        overlap = 0
    return overlap


def follow_iou(gt_mask, mask):

    iou, pmask = Giou_np(mask, gt_mask)
    return iou, pmask
