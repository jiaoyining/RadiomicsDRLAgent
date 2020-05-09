import numpy as np
import cv2


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

    iou, p_mask = calculate_iou(mask, gt_mask)
    return iou, p_mask
