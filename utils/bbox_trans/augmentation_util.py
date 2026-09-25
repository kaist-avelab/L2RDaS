# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import numba
import numpy as np
from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings('ignore', category=NumbaPerformanceWarning)

@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    """Convert boxes_corner to aligned (min-max) boxes.

    Args:
        boxes_corner (np.ndarray, shape=[N, 2**dim, dim]): Boxes corners.

    Returns:
        np.ndarray, shape=[N, dim*2]: Aligned (min-max) boxes.
    """
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result

@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    """Box collision test.

    Args:
        boxes (np.ndarray): Corners of current boxes.
        qboxes (np.ndarray): Boxes to be avoid colliding.
        clockwise (bool, optional): Whether the corners are in
            clockwise order. Default: True.
    """
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, slices, :]),
                           axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = corner_to_standup_nd_jit(boxes)
    qboxes_standup = corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (
                min(boxes_standup[i, 2], qboxes_standup[j, 2]) -
                max(boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (
                    min(boxes_standup[i, 3], qboxes_standup[j, 3]) -
                    max(boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for box_l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, box_l, 0]
                            D = lines_qboxes[j, box_l, 1]
                            acd = (D[1] - A[1]) * (C[0] -
                                                   A[0]) > (C[1] - A[1]) * (
                                                       D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] -
                                                   B[0]) > (C[1] - B[1]) * (
                                                       D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (
                                        C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (
                                        D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for box_l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                    boxes[i, k, 0] - qboxes[j, box_l, 0])
                                cross -= vec[0] * (
                                    boxes[i, k, 1] - qboxes[j, box_l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for box_l in range(4):  # point box_l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                        qboxes[j, k, 0] - boxes[i, box_l, 0])
                                    cross -= vec[0] * (
                                        qboxes[j, k, 1] - boxes[i, box_l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret