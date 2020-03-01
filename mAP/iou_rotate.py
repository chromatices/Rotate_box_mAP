# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from rbbox_overlaps import rbbx_overlaps

def iou_rotate_calculate(boxes1, boxes2, use_gpu=True, gpu_id=0):

    # start = time.time()
    if use_gpu:
        ious = rbbx_overlaps(boxes1, boxes2, gpu_id)
    else:
        area1 = boxes1[:, 2] * boxes1[:, 3]
        area2 = boxes2[:, 2] * boxes2[:, 3]
        ious = []
        for i, box1 in enumerate(boxes1):
            temp_ious = []
            r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
            for j, box2 in enumerate(boxes2):
                r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])
                int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)
                    int_area = cv2.contourArea(order_pts)
                    inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                    temp_ious.append(inter)
                else:
                    temp_ious.append(0.0)
            ious.append(temp_ious)
    return np.array(ious, dtype=np.float32)[0][0]


if __name__ == '__main__':
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = '13'
    boxes1 = np.array([[10, 10, 20, 30, 0]], np.float32)
    boxes2 = np.array([[10, 10, 20, 30, 90]], np.float32)
    ex_ious = iou_rotate_calculate(boxes1, boxes2, use_gpu=False)
    print(ex_ious)
