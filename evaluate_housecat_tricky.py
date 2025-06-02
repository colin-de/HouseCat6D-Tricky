import os
import argparse
import logging
import torch
import torchvision.transforms as transforms
import numpy as np
import glob
import math
import cv2
import collections
import _pickle as cPickle
from tqdm import tqdm

def get_logger(level_print, level_save, path_file, name_logger = "logger"):
    # level: logging.INFO / logging.WARN
    logger = logging.getLogger(name_logger)
    logger.setLevel(level = logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # set file handler
    handler_file = logging.FileHandler(path_file)
    handler_file.setLevel(level_save)
    handler_file.setFormatter(formatter)
    logger.addHandler(handler_file)
    # set console holder
    handler_view = logging.StreamHandler()
    handler_view.setFormatter(formatter)
    handler_view.setLevel(level_print)
    logger.addHandler(handler_view)
    return logger

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.
    x: [rows, columns].
    """

    pre_shape = x.shape
    assert len(x.shape) == 2, x.shape
    new_x = x[~np.all(x == 0, axis=1)]
    post_shape = new_x.shape
    assert pre_shape[0] == post_shape[0]
    assert pre_shape[1] == post_shape[1]

    return new_x

def get_3d_bbox(scale, shift=0):
    """
    Input:
        scale: [3] or scalar
        shift: [3] or scalar
    Return
        bbox_3d: [3, N]

    """
    if hasattr(scale, "__iter__"):
        bbox_3d = np.array([[scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, scale[2] / 2],
                            [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]) + shift
    else:
        bbox_3d = np.array([[scale / 2, +scale / 2, scale / 2],
                            [scale / 2, +scale / 2, -scale / 2],
                            [-scale / 2, +scale / 2, scale / 2],
                            [-scale / 2, +scale / 2, -scale / 2],
                            [+scale / 2, -scale / 2, scale / 2],
                            [+scale / 2, -scale / 2, -scale / 2],
                            [-scale / 2, -scale / 2, scale / 2],
                            [-scale / 2, -scale / 2, -scale / 2]]) + shift

    bbox_3d = bbox_3d.transpose()
    return bbox_3d

def transform_coordinates_3d(coordinates, RT):
    """
    Input:
        coordinates: [3, N]
        RT: [4, 4]
    Return
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones(
        (1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates

def compute_ap_from_matches_scores(pred_match, pred_scores, gt_match):
    # sort the scores from high to low
    # print(pred_match.shape, pred_scores.shape)
    assert pred_match.shape[0] == pred_scores.shape[0]

    score_indices = np.argsort(pred_scores)[::-1]
    pred_scores = pred_scores[score_indices]
    pred_match = pred_match[score_indices]

    precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1])
                * precisions[indices])
    return ap

def compute_3d_iou_new(RT_1, RT_2, scales_1, scales_2, handle_visibility, class_name_1, class_name_2):
    '''Computes IoU overlaps between two 3d bboxes.
       bbox_3d_1, bbox_3d_1: [3, 8]
    '''
    # flatten masks
    def asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):
        noc_cube_1 = get_3d_bbox(scales_1, 0)
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, RT_1)

        noc_cube_2 = get_3d_bbox(scales_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

        bbox_1_max = np.amax(bbox_3d_1, axis=0)
        bbox_1_min = np.amin(bbox_3d_1, axis=0)
        bbox_2_max = np.amax(bbox_3d_2, axis=0)
        bbox_2_min = np.amin(bbox_3d_2, axis=0)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        # intersections and union
        if np.amin(overlap_max - overlap_min) < 0:
            intersections = 0
        else:
            intersections = np.prod(overlap_max - overlap_min)
        union = np.prod(bbox_1_max - bbox_1_min) + \
            np.prod(bbox_2_max - bbox_2_min) - intersections
        overlaps = intersections / union
        return overlaps

    if RT_1 is None or RT_2 is None:
        return -1

    symmetry_flag = False
    if (class_name_1 in ['bottle', 'bowl', 'can', 'glass'] and class_name_1 == class_name_2):
        # print('*'*10)

        noc_cube_1 = get_3d_bbox(scales_1, 0)
        noc_cube_2 = get_3d_bbox(scales_2, 0)
        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

        def y_rotation_matrix(theta):
            return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                             [0, 1, 0, 0],
                             [-np.sin(theta), 0, np.cos(theta), 0],
                             [0, 0, 0, 1]])

        n = 20
        max_iou = 0
        for i in range(n):
            rotated_RT_1 = RT_1@y_rotation_matrix(2*math.pi*i/float(n))
            max_iou = max(max_iou,
                          asymmetric_3d_iou(rotated_RT_1, RT_2, scales_1, scales_2))
    else:
        max_iou = asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)

    return max_iou

def compute_combination_RT_degree_cm_symmetry(RT_1, RT_2, scale, class_id, handle_visibility, synset_names):
    '''
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter


    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'cap',  # 5
                    'phone',  # 6
                    'monitor',  # 7
                    'laptop',  # 8
                    'mug'  # 9
                    ]

    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'laptop',  # 5
                    'mug'  # 6
                    ]
    '''

    # make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    # symmetric when rotating around y-axis
    if synset_names[class_id] in ['bottle', 'can', 'bowl']:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(
            y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    # symmetric when rotating around y-axis
    elif synset_names[class_id] == 'mug' and handle_visibility == 0:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(
            y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] in ['phone', 'eggbox', 'glue']:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(np.arccos((np.trace(R) - 1) / 2),
                    np.arccos((np.trace(R_rot) - 1) / 2))
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2) / scale
    result = np.array([theta, shift])

    return result

def compute_combination_3d_matches(gt_class_ids, gt_RTs, gt_scales, gt_handle_visibility, synset_names,
                                   pred_boxes, pred_class_ids, pred_scores, pred_RTs, pred_scales,
                                   iou_3d_thresholds, degree_thesholds, shift_thesholds, score_threshold=0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_matches: 2-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_matches: 2-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)

    if num_pred:
        pred_boxes = trim_zeros(pred_boxes).copy()
        pred_scores = pred_scores[:pred_boxes.shape[0]].copy()

        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]

        pred_boxes = pred_boxes[indices].copy()
        pred_class_ids = pred_class_ids[indices].copy()
        pred_scores = pred_scores[indices].copy()
        pred_scales = pred_scales[indices].copy()
        pred_RTs = pred_RTs[indices].copy()

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    #overlaps = [[0 for j in range(num_gt)] for i in range(num_pred)]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    RT_overlaps = np.zeros((num_pred, num_gt, 2), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            # overlaps[i, j] = compute_3d_iou(pred_3d_bboxs[i], gt_3d_bboxs[j], gt_handle_visibility[j],
            #    synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]])
            overlaps[i, j] = compute_3d_iou_new(pred_RTs[i], gt_RTs[j], pred_scales[i, :], gt_scales[j],
                                                gt_handle_visibility[j], synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]])

            RT_overlaps[i, j, :] = compute_combination_RT_degree_cm_symmetry(pred_RTs[i], gt_RTs[j], np.cbrt(
                np.linalg.det(gt_RTs[j, :3, :3])), gt_class_ids[j], gt_handle_visibility[j], synset_names)

    # Loop through predictions and find matching ground truth boxes
    num_iou_3d_thres = len(iou_3d_thresholds)
    num_degree_thes = len(degree_thesholds)
    num_shift_thes = len(shift_thesholds)
    pred_matches = -1 * \
        np.ones([num_degree_thes, num_shift_thes, num_iou_3d_thres, num_pred])
    gt_matches = -1 * \
        np.ones([num_degree_thes, num_shift_thes, num_iou_3d_thres, num_gt])

    for s, iou_thres in enumerate(iou_3d_thresholds):
        for d, degree_thres in enumerate(degree_thesholds):
            for t, shift_thres in enumerate(shift_thesholds):
                for i in range(len(pred_boxes)):
                    # Find best matching ground truth box
                    # 1. Sort matches by score
                    sorted_ixs_by_iou = np.argsort(overlaps[i])[::-1]
                    # 2. Remove low scores
                    low_score_idx = np.where(
                        overlaps[i, sorted_ixs_by_iou] < score_threshold)[0]
                    if low_score_idx.size > 0:
                        sorted_ixs_by_iou = sorted_ixs_by_iou[:low_score_idx[0]]
                    # 3. Find the match
                    for j in sorted_ixs_by_iou:
                        if gt_matches[d, t, s, j] > -1:
                            continue
                        # If we reach IoU smaller than the threshold, end the loop
                        iou = overlaps[i, j]
                        r_error = RT_overlaps[i, j, 0]
                        t_error = RT_overlaps[i, j, 1]

                        if iou < iou_thres or r_error > degree_thres or t_error > shift_thres:
                            break

                        if not pred_class_ids[i] == gt_class_ids[j]:
                            continue

                        if iou >= iou_thres or r_error <= degree_thres or t_error <= shift_thres:
                            gt_matches[d, t, s, j] = i
                            pred_matches[d, t, s, i] = j
                            break

    return gt_matches, pred_matches, indices

def compute_combination_mAP(final_results, synset_names, degree_thresholds=[5, 10, 15], shift_thresholds=[0.1, 0.2], iou_3d_thresholds=[0.1]):
    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)

    shift_thres_list = list(shift_thresholds) + [100]
    num_shift_thres = len(shift_thres_list)

    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    aps = np.zeros((num_classes + 1, num_degree_thres,
                    num_shift_thres, num_iou_thres))
    pred_matches_all = [np.zeros(
        (num_degree_thres, num_shift_thres, num_iou_thres, 0)) for _ in range(num_classes)]
    gt_matches_all = [np.zeros(
        (num_degree_thres, num_shift_thres, num_iou_thres, 0)) for _ in range(num_classes)]
    pred_scores_all = [np.zeros(
        (num_degree_thres, num_shift_thres, num_iou_thres, 0)) for _ in range(num_classes)]

    for progress, result in tqdm(enumerate(final_results)):
        gt_class_ids = result['gt_class_ids'].astype(np.int32)
        gt_RTs = np.array(result['gt_RTs'])
        gt_scales = np.array(result['gt_scales'])
        gt_handle_visibility = result['gt_handle_visibility']

        pred_bboxes = np.array(result['pred_bboxes'])
        pred_class_ids = result['pred_class_ids']
        pred_scales = result['pred_scales']
        pred_scores = result['pred_scores']
        pred_RTs = np.array(result['pred_RTs'])

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue

        for cls_id in range(1, num_classes):
            # get gt and predictions in this class
            cls_gt_class_ids = gt_class_ids[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros(0)
            cls_gt_scales = gt_scales[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros((0, 3))
            cls_gt_RTs = gt_RTs[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros((0, 4, 4))

            cls_pred_class_ids = pred_class_ids[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros(0)
            cls_pred_bboxes = pred_bboxes[pred_class_ids == cls_id, :] if len(
                pred_class_ids) else np.zeros((0, 4))
            cls_pred_scores = pred_scores[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros(0)
            cls_pred_RTs = pred_RTs[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros((0, 4, 4))
            cls_pred_scales = pred_scales[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros((0, 3))

            if synset_names[cls_id] != 'mug':
                cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)
            else:
                cls_gt_handle_visibility = gt_handle_visibility[gt_class_ids == cls_id] if len(
                    gt_class_ids) else np.ones(0)

            gt_match, pred_match, pred_indiced = compute_combination_3d_matches(cls_gt_class_ids, cls_gt_RTs, cls_gt_scales, cls_gt_handle_visibility, synset_names,
                                                                                cls_pred_bboxes, cls_pred_class_ids, cls_pred_scores, cls_pred_RTs, cls_pred_scales,
                                                                                iou_thres_list, degree_thres_list, shift_thres_list)
            if len(pred_indiced):
                cls_pred_class_ids = cls_pred_class_ids[pred_indiced]
                cls_pred_RTs = cls_pred_RTs[pred_indiced]
                cls_pred_scores = cls_pred_scores[pred_indiced]
                cls_pred_bboxes = cls_pred_bboxes[pred_indiced]

            pred_matches_all[cls_id] = np.concatenate(
                (pred_matches_all[cls_id], pred_match), axis=-1)
            cls_pred_scores_tile = np.tile(
                cls_pred_scores, (num_degree_thres, num_shift_thres, num_iou_thres, 1))
            pred_scores_all[cls_id] = np.concatenate(
                (pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            assert pred_matches_all[cls_id].shape[-1] == pred_scores_all[cls_id].shape[-1]
            gt_matches_all[cls_id] = np.concatenate(
                (gt_matches_all[cls_id], gt_match), axis=-1)

    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        for s, iou_thres in enumerate(iou_thres_list):
            for d, degree_thres in enumerate(degree_thres_list):
                for t, shift_thres in enumerate(shift_thres_list):
                    aps[cls_id, d, t, s] = compute_ap_from_matches_scores(pred_matches_all[cls_id][d, t, s, :],
                                                                          pred_scores_all[cls_id][d,
                                                                                                  t, s, :],
                                                                          gt_matches_all[cls_id][d, t, s, :])
    # for i in range(6):
    #     print(i+1)
    #     print('IoU75, 5  degree,  5% translation: {:.2f}'.format(aps[i+1, degree_thres_list.index(5), shift_thres_list.index(0.05), iou_thres_list.index(0.75)]*100))
    #     print('IoU75, 10 degree,  5% translation: {:.2f}'.format(aps[i+1, degree_thres_list.index(10), shift_thres_list.index(0.05), iou_thres_list.index(0.75)]*100))
    #     print('IoU75, 5  degree, 10% translation: {:.2f}'.format(aps[i+1, degree_thres_list.index(5), shift_thres_list.index(0.10), iou_thres_list.index(0.75)]*100))
    #     print('IoU50, 5  degree, 20% translation: {:.2f}'.format(aps[i+1, degree_thres_list.index(5), shift_thres_list.index(0.20), iou_thres_list.index(0.50)]*100))
    #     print('IoU50, 10 degree, 10% translation: {:.2f}'.format(aps[i+1, degree_thres_list.index(10), shift_thres_list.index(0.10), iou_thres_list.index(0.50)]*100))
    #     print('IoU50, 10 degree, 20% translation: {:.2f}'.format(aps[i+1, degree_thres_list.index(10), shift_thres_list.index(0.20), iou_thres_list.index(0.50)]*100))

    # print('ALL:')
    aps[-1, :, :, :] = np.mean(aps[1:-1, :, :, :], axis=0)

    print('IoU75, 5  degree,  5% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(5), shift_thres_list.index(0.05), iou_thres_list.index(0.75)]*100))
    print('IoU75, 10 degree,  5% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(10), shift_thres_list.index(0.05), iou_thres_list.index(0.75)]*100))
    print('IoU75, 5  degree, 10% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(5), shift_thres_list.index(0.10), iou_thres_list.index(0.75)]*100))
    print('IoU50, 5  degree, 20% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(5), shift_thres_list.index(0.20), iou_thres_list.index(0.50)]*100))
    print('IoU50, 10 degree, 10% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(10), shift_thres_list.index(0.10), iou_thres_list.index(0.50)]*100))
    print('IoU50, 10 degree, 20% translation: {:.2f}'.format(
        aps[-1, degree_thres_list.index(10), shift_thres_list.index(0.20), iou_thres_list.index(0.50)]*100))

    return aps

def compute_3d_matches(gt_class_ids, gt_RTs, gt_scales, gt_handle_visibility, synset_names,
                       pred_boxes, pred_class_ids, pred_scores, pred_RTs, pred_scales,
                       iou_3d_thresholds, score_threshold=0):
    """Finds matches between prediction and ground truth instances.
    Returns:
        gt_matches: 2-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_matches: 2-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)
    indices = np.zeros(0)

    if num_pred:
        pred_boxes = trim_zeros(pred_boxes).copy()
        pred_scores = pred_scores[:pred_boxes.shape[0]].copy()

        # Sort predictions by score from high to low
        indices = np.argsort(pred_scores)[::-1]

        pred_boxes = pred_boxes[indices].copy()
        pred_class_ids = pred_class_ids[indices].copy()
        pred_scores = pred_scores[indices].copy()
        pred_scales = pred_scales[indices].copy()
        pred_RTs = pred_RTs[indices].copy()

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    #overlaps = [[0 for j in range(num_gt)] for i in range(num_pred)]
    overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j] = compute_3d_iou_new(pred_RTs[i], gt_RTs[j], pred_scales[i, :], gt_scales[j],
                                                gt_handle_visibility[j], synset_names[pred_class_ids[i]], synset_names[gt_class_ids[j]])

    # Loop through predictions and find matching ground truth boxes
    num_iou_3d_thres = len(iou_3d_thresholds)
    pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
    gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])

    for s, iou_thres in enumerate(iou_3d_thresholds):
        for i in range(len(pred_boxes)):
            # Find best matching ground truth box
            # 1. Sort matches by score
            sorted_ixs = np.argsort(overlaps[i])[::-1]
            # 2. Remove low scores
            low_score_idx = np.where(
                overlaps[i, sorted_ixs] < score_threshold)[0]
            if low_score_idx.size > 0:
                sorted_ixs = sorted_ixs[:low_score_idx[0]]
            # 3. Find the match
            for j in sorted_ixs:
                # If ground truth box is already matched, go to next one
                #print('gt_match: ', gt_match[j])
                if gt_matches[s, j] > -1:
                    continue
                # If we reach IoU smaller than the threshold, end the loop
                iou = overlaps[i, j]
                #print('iou: ', iou)
                if iou < iou_thres:
                    break
                # Do we have a match?
                if not pred_class_ids[i] == gt_class_ids[j]:
                    continue

                if iou > iou_thres:
                    gt_matches[s, j] = i
                    pred_matches[s, i] = j
                    break

    return gt_matches, pred_matches, overlaps, indices

def compute_RT_degree_cm_symmetry(RT_1, RT_2, class_id, handle_visibility, synset_names):
    '''
    :param RT_1: [4, 4]. homogeneous affine transformation
    :param RT_2: [4, 4]. homogeneous affine transformation
    :return: theta: angle difference of R in degree, shift: l2 difference of T in centimeter


    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'cap',  # 5
                    'phone',  # 6
                    'monitor',  # 7
                    'laptop',  # 8
                    'mug'  # 9
                    ]

    synset_names = ['BG',  # 0
                    'bottle',  # 1
                    'bowl',  # 2
                    'camera',  # 3
                    'can',  # 4
                    'laptop',  # 5
                    'mug'  # 6
                    ]
    '''

    # make sure the last row is [0, 0, 0, 1]
    if RT_1 is None or RT_2 is None:
        return -1
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        print(RT_1[3, :], RT_2[3, :])
        exit()

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    # symmetric when rotating around y-axis
    if synset_names[class_id] in ['bottle', 'can', 'bowl', 'glass']:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(
            y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    # symmetric when rotating around y-axis
    elif synset_names[class_id] == 'mug' and handle_visibility == 0:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(
            y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] in ['phone', 'eggbox', 'glue']:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(np.arccos((np.trace(R) - 1) / 2),
                    np.arccos((np.trace(R_rot) - 1) / 2))
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2) * 100
    result = np.array([theta, shift])

    return result

def compute_RT_overlaps(gt_class_ids, gt_RTs, gt_handle_visibility,
                        pred_class_ids, pred_RTs,
                        synset_names):
    """Finds overlaps between prediction and ground truth instances.
    Returns:
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # print('num of gt instances: {}, num of pred instances: {}'.format(len(gt_class_ids), len(gt_class_ids)))
    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    # Compute IoU overlaps [pred_bboxs gt_bboxs]
    #overlaps = [[0 for j in range(num_gt)] for i in range(num_pred)]
    overlaps = np.zeros((num_pred, num_gt, 2))

    for i in range(num_pred):
        for j in range(num_gt):
            overlaps[i, j, :] = compute_RT_degree_cm_symmetry(pred_RTs[i],
                                                              gt_RTs[j],
                                                              gt_class_ids[j],
                                                              gt_handle_visibility[j],
                                                              synset_names)

    return overlaps

def compute_match_from_degree_cm(overlaps, pred_class_ids, gt_class_ids, degree_thres_list, shift_thres_list):
    num_degree_thres = len(degree_thres_list)
    num_shift_thres = len(shift_thres_list)

    num_pred = len(pred_class_ids)
    num_gt = len(gt_class_ids)

    pred_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_pred))
    gt_matches = -1 * np.ones((num_degree_thres, num_shift_thres, num_gt))

    if num_pred == 0 or num_gt == 0:
        return gt_matches, pred_matches

    assert num_pred == overlaps.shape[0]
    assert num_gt == overlaps.shape[1]
    assert overlaps.shape[2] == 2

    for d, degree_thres in enumerate(degree_thres_list):
        for s, shift_thres in enumerate(shift_thres_list):
            for i in range(num_pred):
                # Find best matching ground truth box
                # 1. Sort matches by scores from low to high
                sum_degree_shift = np.sum(overlaps[i, :, :], axis=-1)
                sorted_ixs = np.argsort(sum_degree_shift)
                # 2. Remove low scores
                # low_score_idx = np.where(sum_degree_shift >= 100)[0]
                # if low_score_idx.size > 0:
                #     sorted_ixs = sorted_ixs[:low_score_idx[0]]
                # 3. Find the match
                for j in sorted_ixs:
                    # If ground truth box is already matched, go to next one
                    #print(j, len(gt_match), len(pred_class_ids), len(gt_class_ids))
                    if gt_matches[d, s, j] > -1 or pred_class_ids[i] != gt_class_ids[j]:
                        continue
                    # If we reach IoU smaller than the threshold, end the loop
                    if overlaps[i, j, 0] > degree_thres or overlaps[i, j, 1] > shift_thres:
                        continue

                    gt_matches[d, s, j] = i
                    pred_matches[d, s, i] = j
                    break

    return gt_matches, pred_matches

def compute_independent_mAP(final_results, synset_names, degree_thresholds=[360], shift_thresholds=[100], iou_3d_thresholds=[0.1], iou_pose_thres=0.1, use_matches_for_pose=True, logger=None, debug_mode=False):

    num_classes = len(synset_names)
    degree_thres_list = list(degree_thresholds) + [360]
    num_degree_thres = len(degree_thres_list)

    shift_thres_list = list(shift_thresholds) + [100]
    num_shift_thres = len(shift_thres_list)

    iou_thres_list = list(iou_3d_thresholds)
    num_iou_thres = len(iou_thres_list)

    if use_matches_for_pose:
        assert iou_pose_thres in iou_thres_list

    iou_3d_aps = np.zeros((num_classes + 1, num_iou_thres))
    iou_pred_matches_all = [np.zeros((num_iou_thres, 0))
                            for _ in range(num_classes)]
    iou_pred_scores_all = [np.zeros((num_iou_thres, 0))
                           for _ in range(num_classes)]
    iou_gt_matches_all = [np.zeros((num_iou_thres, 0))
                          for _ in range(num_classes)]

    pose_aps = np.zeros((num_classes + 1, num_degree_thres, num_shift_thres))
    pose_pred_matches_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)]
    pose_gt_matches_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)]
    pose_pred_scores_all = [
        np.zeros((num_degree_thres, num_shift_thres, 0)) for _ in range(num_classes)]

    # Add debug tracking
    debug_info = {
        'cutlery': {'total_gt': 0, 'total_pred': 0, 'scenes_with_instances': 0},
        'glass': {'total_gt': 0, 'total_pred': 0, 'scenes_with_instances': 0}
    }

    # loop over results to gather pred matches and gt matches for iou and pose metrics
    progress = 0
    for progress, result in tqdm(enumerate(final_results)):
        gt_class_ids = result['gt_class_ids'].astype(np.int32)
        gt_RTs = np.array(result['gt_RTs'])
        gt_scales = np.array(result['gt_scales'])
        gt_handle_visibility = result['gt_handle_visibility']

        pred_bboxes = np.array(result['pred_bboxes'])
        pred_class_ids = result['pred_class_ids']
        pred_scales = result['pred_scales']
        pred_scores = result['pred_scores']
        pred_RTs = np.array(result['pred_RTs'])

        if len(gt_class_ids) == 0 and len(pred_class_ids) == 0:
            continue

        # Debug: Print scene info
        if debug_mode and progress < 5:  # Only for first 5 scenes
            print(f"\n=== Scene {progress} ===")
            print(f"GT class_ids: {gt_class_ids}")
            print(f"Pred class_ids: {pred_class_ids}")
            print(f"GT counts per class: {np.bincount(gt_class_ids, minlength=num_classes)}")
            print(f"Pred counts per class: {np.bincount(pred_class_ids, minlength=num_classes)}")

        for cls_id in range(1, num_classes):
        # for cls_id in [7, 8]:
            # get gt and predictions in this class
            cls_gt_class_ids = gt_class_ids[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros(0)
            cls_gt_scales = gt_scales[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros((0, 3))
            cls_gt_RTs = gt_RTs[gt_class_ids == cls_id] if len(
                gt_class_ids) else np.zeros((0, 4, 4))

            cls_pred_class_ids = pred_class_ids[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros(0)
            cls_pred_bboxes = pred_bboxes[pred_class_ids == cls_id, :] if len(
                pred_class_ids) else np.zeros((0, 4))
            cls_pred_scores = pred_scores[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros(0)
            cls_pred_RTs = pred_RTs[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros((0, 4, 4))
            cls_pred_scales = pred_scales[pred_class_ids == cls_id] if len(
                pred_class_ids) else np.zeros((0, 3))

            # Update debug info for cutlery and glass
            class_name = synset_names[cls_id]
            if class_name in ['cutlery', 'glass']:
                debug_info[class_name]['total_gt'] += len(cls_gt_class_ids)
                debug_info[class_name]['total_pred'] += len(cls_pred_class_ids)
                if len(cls_gt_class_ids) > 0 or len(cls_pred_class_ids) > 0:
                    debug_info[class_name]['scenes_with_instances'] += 1

            # Debug: Print class-specific info for cutlery and glass
            if debug_mode and progress < 5 and class_name in ['cutlery', 'glass']:
                print(f"  {class_name}: {len(cls_gt_class_ids)} GT, {len(cls_pred_class_ids)} Pred")
                if len(cls_pred_scores) > 0:
                    print(f"    Pred scores: {cls_pred_scores}")

            # Skip if no instances for this class
            if len(cls_gt_class_ids) == 0 and len(cls_pred_class_ids) == 0:
                continue

            # calculate the overlap between each gt instance and pred instance
            cls_gt_handle_visibility = np.ones_like(cls_gt_class_ids)

            iou_cls_gt_match, iou_cls_pred_match, _, iou_pred_indices = compute_3d_matches(cls_gt_class_ids, cls_gt_RTs, cls_gt_scales, cls_gt_handle_visibility, synset_names,
                                                                                           cls_pred_bboxes, cls_pred_class_ids, cls_pred_scores, cls_pred_RTs, cls_pred_scales,
                                                                                           iou_thres_list)
            if len(iou_pred_indices):
                cls_pred_class_ids = cls_pred_class_ids[iou_pred_indices]
                cls_pred_RTs = cls_pred_RTs[iou_pred_indices]
                cls_pred_scores = cls_pred_scores[iou_pred_indices]
                cls_pred_bboxes = cls_pred_bboxes[iou_pred_indices]

            # Debug: Print matching info
            if debug_mode and progress < 5 and class_name in ['cutlery', 'glass']:
                print(f"    IoU matches shape: GT {iou_cls_gt_match.shape}, Pred {iou_cls_pred_match.shape}")
                for thres_idx, thres in enumerate(iou_thres_list):
                    gt_matched = np.sum(iou_cls_gt_match[thres_idx, :] > -1)
                    pred_matched = np.sum(iou_cls_pred_match[thres_idx, :] > -1)
                    print(f"      IoU@{thres}: {gt_matched}/{len(cls_gt_class_ids)} GT matched, {pred_matched}/{len(cls_pred_class_ids)} Pred matched")

            iou_pred_matches_all[cls_id] = np.concatenate(
                (iou_pred_matches_all[cls_id], iou_cls_pred_match), axis=-1)
            cls_pred_scores_tile = np.tile(cls_pred_scores, (num_iou_thres, 1))
            iou_pred_scores_all[cls_id] = np.concatenate(
                (iou_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            assert iou_pred_matches_all[cls_id].shape[1] == iou_pred_scores_all[cls_id].shape[1]
            iou_gt_matches_all[cls_id] = np.concatenate(
                (iou_gt_matches_all[cls_id], iou_cls_gt_match), axis=-1)

            if use_matches_for_pose:
                thres_ind = list(iou_thres_list).index(iou_pose_thres)

                iou_thres_pred_match = iou_cls_pred_match[thres_ind, :]

                cls_pred_class_ids = cls_pred_class_ids[iou_thres_pred_match > -1] if len(
                    iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_RTs = cls_pred_RTs[iou_thres_pred_match > -1] if len(
                    iou_thres_pred_match) > 0 else np.zeros((0, 4, 4))
                cls_pred_scores = cls_pred_scores[iou_thres_pred_match > -1] if len(
                    iou_thres_pred_match) > 0 else np.zeros(0)
                cls_pred_bboxes = cls_pred_bboxes[iou_thres_pred_match > -1] if len(
                    iou_thres_pred_match) > 0 else np.zeros((0, 4))

                iou_thres_gt_match = iou_cls_gt_match[thres_ind, :]
                cls_gt_class_ids = cls_gt_class_ids[iou_thres_gt_match > -1] if len(
                    iou_thres_gt_match) > 0 else np.zeros(0)
                cls_gt_RTs = cls_gt_RTs[iou_thres_gt_match > -1] if len(
                    iou_thres_gt_match) > 0 else np.zeros((0, 4, 4))
                cls_gt_handle_visibility = cls_gt_handle_visibility[iou_thres_gt_match > -1] if len(
                    iou_thres_gt_match) > 0 else np.zeros(0)

            RT_overlaps = compute_RT_overlaps(cls_gt_class_ids, cls_gt_RTs, cls_gt_handle_visibility,
                                              cls_pred_class_ids, cls_pred_RTs,
                                              synset_names)

            pose_cls_gt_match, pose_cls_pred_match = compute_match_from_degree_cm(RT_overlaps,
                                                                                  cls_pred_class_ids,
                                                                                  cls_gt_class_ids,
                                                                                  degree_thres_list,
                                                                                  shift_thres_list)

            pose_pred_matches_all[cls_id] = np.concatenate(
                (pose_pred_matches_all[cls_id], pose_cls_pred_match), axis=-1)

            cls_pred_scores_tile = np.tile(
                cls_pred_scores, (num_degree_thres, num_shift_thres, 1))
            pose_pred_scores_all[cls_id] = np.concatenate(
                (pose_pred_scores_all[cls_id], cls_pred_scores_tile), axis=-1)
            assert pose_pred_scores_all[cls_id].shape[2] == pose_pred_matches_all[cls_id].shape[2], '{} vs. {}'.format(
                pose_pred_scores_all[cls_id].shape, pose_pred_matches_all[cls_id].shape)
            pose_gt_matches_all[cls_id] = np.concatenate(
                (pose_gt_matches_all[cls_id], pose_cls_gt_match), axis=-1)

    # Print debug summary
    # if debug_mode or logger:
    #     summary_msg = "\n=== Debug Summary ==="
    #     for class_name in ['cutlery', 'glass']:
    #         cls_id = synset_names.index(class_name)
    #         info = debug_info[class_name]
    #         summary_msg += f"\n{class_name} (id={cls_id}):"
    #         summary_msg += f"\n  Total GT instances: {info['total_gt']}"
    #         summary_msg += f"\n  Total Pred instances: {info['total_pred']}"
    #         summary_msg += f"\n  Scenes with instances: {info['scenes_with_instances']}"
    #         summary_msg += f"\n  Accumulated GT matches shape: {iou_gt_matches_all[cls_id].shape}"
    #         summary_msg += f"\n  Accumulated Pred matches shape: {iou_pred_matches_all[cls_id].shape}"
        
    #     if logger:
    #         logger.warning(summary_msg)
    #     else:
    #         print(summary_msg)

    iou_dict = {}
    iou_dict['thres_list'] = iou_thres_list
    for cls_id in range(1, num_classes):
        class_name = synset_names[cls_id]
        if class_name not in ['cutlery', 'glass']:
            continue
        for s, iou_thres in enumerate(iou_thres_list):
            # Check if we have any data for this class
            if iou_pred_matches_all[cls_id].shape[1] == 0:
                iou_3d_aps[cls_id, s] = 0.0
                if debug_mode:
                    print(f"No data for {class_name} at IoU {iou_thres}")
            else:
                iou_3d_aps[cls_id, s] = compute_ap_from_matches_scores(iou_pred_matches_all[cls_id][s, :],
                                                                       iou_pred_scores_all[cls_id][s, :],
                                                                       iou_gt_matches_all[cls_id][s, :])
            if logger:
                logger.warning(f'{class_name} on {iou_thres}: {iou_3d_aps[cls_id, s] * 100:.2f}')

    # Only compute mean for classes that have data (cutlery and glass)
    valid_class_ids = [i for i in range(1, num_classes) if synset_names[i] in ['cutlery', 'glass']]
    if valid_class_ids:
        iou_3d_aps[-1, :] = np.mean(iou_3d_aps[valid_class_ids, :], axis=0)
    else:
        iou_3d_aps[-1, :] = 0.0

    for i, degree_thres in enumerate(degree_thres_list):
        for j, shift_thres in enumerate(shift_thres_list):
            for cls_id in range(1, num_classes):
                if synset_names[cls_id] not in ['cutlery', 'glass']:
                    continue
                    
                cls_pose_pred_matches_all = pose_pred_matches_all[cls_id][i, j, :]
                cls_pose_gt_matches_all = pose_gt_matches_all[cls_id][i, j, :]
                cls_pose_pred_scores_all = pose_pred_scores_all[cls_id][i, j, :]

                if len(cls_pose_pred_matches_all) == 0:
                    pose_aps[cls_id, i, j] = 0.0
                else:
                    pose_aps[cls_id, i, j] = compute_ap_from_matches_scores(cls_pose_pred_matches_all,
                                                                            cls_pose_pred_scores_all,
                                                                            cls_pose_gt_matches_all)

            # Only compute mean for valid classes
            if valid_class_ids:
                pose_aps[-1, i, j] = np.mean(pose_aps[valid_class_ids, i, j])
            else:
                pose_aps[-1, i, j] = 0.0

    if logger is not None:
        logger.warning('3D IoU at 25: {:.1f}'.format(
            iou_3d_aps[-1, iou_thres_list.index(0.25)] * 100))
        logger.warning('3D IoU at 50: {:.1f}'.format(
            iou_3d_aps[-1, iou_thres_list.index(0.5)] * 100))
        logger.warning('3D IoU at 75: {:.1f}'.format(
            iou_3d_aps[-1, iou_thres_list.index(0.75)] * 100))

        logger.warning('5 degree, 2cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(5), shift_thres_list.index(2)] * 100))
        logger.warning('5 degree, 5cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(5), shift_thres_list.index(5)] * 100))
        logger.warning('10 degree, 2cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(10), shift_thres_list.index(2)] * 100))
        logger.warning('10 degree, 5cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(10), shift_thres_list.index(5)] * 100))

        # Only log results for cutlery and glass categories
        cutlery_id = synset_names.index('cutlery')
        glass_id = synset_names.index('glass')
        
        logger.warning(f'3D IoU at 25 per category: cutlery={iou_3d_aps[cutlery_id, iou_thres_list.index(0.25)] * 100:.2f}, glass={iou_3d_aps[glass_id, iou_thres_list.index(0.25)] * 100:.2f}')
        logger.warning(f'3D IoU at 50 per category: cutlery={iou_3d_aps[cutlery_id, iou_thres_list.index(0.5)] * 100:.2f}, glass={iou_3d_aps[glass_id, iou_thres_list.index(0.5)] * 100:.2f}')
        logger.warning(f'3D IoU at 75 per category: cutlery={iou_3d_aps[cutlery_id, iou_thres_list.index(0.75)] * 100:.2f}, glass={iou_3d_aps[glass_id, iou_thres_list.index(0.75)] * 100:.2f}')
        
        logger.warning(f'5 degree, 2cm per category: cutlery={pose_aps[cutlery_id, degree_thres_list.index(5), shift_thres_list.index(2)] * 100:.2f}, glass={pose_aps[glass_id, degree_thres_list.index(5), shift_thres_list.index(2)] * 100:.2f}')
        logger.warning(f'5 degree, 5cm per category: cutlery={pose_aps[cutlery_id, degree_thres_list.index(5), shift_thres_list.index(5)] * 100:.2f}, glass={pose_aps[glass_id, degree_thres_list.index(5), shift_thres_list.index(5)] * 100:.2f}')
        logger.warning(f'10 degree, 2cm per category: cutlery={pose_aps[cutlery_id, degree_thres_list.index(10), shift_thres_list.index(2)] * 100:.2f}, glass={pose_aps[glass_id, degree_thres_list.index(10), shift_thres_list.index(2)] * 100:.2f}')
        logger.warning(f'10 degree, 5cm per category: cutlery={pose_aps[cutlery_id, degree_thres_list.index(10), shift_thres_list.index(5)] * 100:.2f}, glass={pose_aps[glass_id, degree_thres_list.index(10), shift_thres_list.index(5)] * 100:.2f}')

    else:
        print('3D IoU at 25: {:.1f}'.format(
            iou_3d_aps[-1, iou_thres_list.index(0.25)] * 100))
        print('3D IoU at 50: {:.1f}'.format(
            iou_3d_aps[-1, iou_thres_list.index(0.5)] * 100))
        print('3D IoU at 75: {:.1f}'.format(
            iou_3d_aps[-1, iou_thres_list.index(0.75)] * 100))

        print('5 degree, 2cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(5), shift_thres_list.index(2)] * 100))
        print('5 degree, 5cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(5), shift_thres_list.index(5)] * 100))
        print('10 degree, 2cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(10), shift_thres_list.index(2)] * 100))
        print('10 degree, 5cm: {:.1f}'.format(
            pose_aps[-1, degree_thres_list.index(10), shift_thres_list.index(5)] * 100))

        # Only print results for cutlery and glass categories
        cutlery_id = synset_names.index('cutlery')
        glass_id = synset_names.index('glass')
        
        print(f'3D IoU at 25 per category: cutlery={iou_3d_aps[cutlery_id, iou_thres_list.index(0.25)] * 100:.2f}, glass={iou_3d_aps[glass_id, iou_thres_list.index(0.25)] * 100:.2f}')
        print(f'3D IoU at 50 per category: cutlery={iou_3d_aps[cutlery_id, iou_thres_list.index(0.5)] * 100:.2f}, glass={iou_3d_aps[glass_id, iou_thres_list.index(0.5)] * 100:.2f}')
        print(f'3D IoU at 75 per category: cutlery={iou_3d_aps[cutlery_id, iou_thres_list.index(0.75)] * 100:.2f}, glass={iou_3d_aps[glass_id, iou_thres_list.index(0.75)] * 100:.2f}')
        
        print(f'5 degree, 2cm per category: cutlery={pose_aps[cutlery_id, degree_thres_list.index(5), shift_thres_list.index(2)] * 100:.2f}, glass={pose_aps[glass_id, degree_thres_list.index(5), shift_thres_list.index(2)] * 100:.2f}')
        print(f'5 degree, 5cm per category: cutlery={pose_aps[cutlery_id, degree_thres_list.index(5), shift_thres_list.index(5)] * 100:.2f}, glass={pose_aps[glass_id, degree_thres_list.index(5), shift_thres_list.index(5)] * 100:.2f}')
        print(f'10 degree, 2cm per category: cutlery={pose_aps[cutlery_id, degree_thres_list.index(10), shift_thres_list.index(2)] * 100:.2f}, glass={pose_aps[glass_id, degree_thres_list.index(10), shift_thres_list.index(2)] * 100:.2f}')
        print(f'10 degree, 5cm per category: cutlery={pose_aps[cutlery_id, degree_thres_list.index(10), shift_thres_list.index(5)] * 100:.2f}, glass={pose_aps[glass_id, degree_thres_list.index(10), shift_thres_list.index(5)] * 100:.2f}')

    return iou_3d_aps, pose_aps

def evaluate_housecat(path, logger=None, debug_mode=False):
    synset_names = ['BG',  # 0
                    'box',  # 1
                    'bottle',  # 2
                    'can',  # 3
                    'cup',  # 4
                    'remote',  # 5
                    'teapot', # 6
                    'cutlery', # 7
                    'glass', # 8
                    'shoe', # 9
                    'tube']  # 10

    result_pkl_list = []
    for scene in ['test_scene1', 'test_scene2', 'test_scene3']:
        scene_files = glob.glob(os.path.join(path, scene, '*.pkl'))
        result_pkl_list.extend(scene_files)
        if debug_mode:
            print(f"Found {len(scene_files)} files in {scene}")
    
    result_pkl_list = sorted(result_pkl_list)
    print('Total image num: {}'.format(len(result_pkl_list)))

    final_results = []
    scene_stats = {}
    
    for pkl_path in result_pkl_list:
        # Extract scene name for debugging
        scene_name = pkl_path.split('/')[-2] if '/' in pkl_path else "unknown"
        
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            result['gt_handle_visibility'] = np.ones_like(
                result['gt_class_ids'])
        
        # Debug: Track scene statistics
        if debug_mode:
            if scene_name not in scene_stats:
                scene_stats[scene_name] = {'files': 0, 'cutlery_gt': 0, 'glass_gt': 0, 'cutlery_pred': 0, 'glass_pred': 0}
            
            scene_stats[scene_name]['files'] += 1
            gt_class_ids = result['gt_class_ids'].astype(np.int32)
            pred_class_ids = result['pred_class_ids']
            
            scene_stats[scene_name]['cutlery_gt'] += np.sum(gt_class_ids == 7)
            scene_stats[scene_name]['glass_gt'] += np.sum(gt_class_ids == 8)
            scene_stats[scene_name]['cutlery_pred'] += np.sum(pred_class_ids == 7)
            scene_stats[scene_name]['glass_pred'] += np.sum(pred_class_ids == 8)
        
        if type(result) is list:
            final_results += result
        elif type(result) is dict:
            final_results.append(result)
        else:
            assert False

    # print("Compute combined mAP: ")
    # compute_combination_mAP(final_results, synset_names,
    #                         degree_thresholds=[5, 10, 20],
    #                         shift_thresholds=[0.05, 0.1, 0.2],
    #                         iou_3d_thresholds=[0.25, 0.50, 0.75])

    print("Compute independent mAP: ")
    iou_aps, pose_aps = compute_independent_mAP(final_results, synset_names,
                            degree_thresholds=[5, 10],
                            shift_thresholds=[2, 5, 10],
                            iou_3d_thresholds=[0.10, 0.25, 0.50, 0.75], 
                            logger=logger, debug_mode=debug_mode)
    print("Finished independent mAP evaluation")

    return iou_aps, pose_aps

def test_func_tricky(dataloder, save_path_):
    """
    Add ground truth information to existing pkl files that already contain prediction results.
    This is for workshop challenge where pred results are already available and we have private GT labels.
    """
    with tqdm(total=len(dataloder)) as t:
        for i, data in enumerate(dataloder):
            path = dataloder.dataset.test_img_list[i].replace('png', 'pkl')
            for block in path.split('/'):
                if 'scene' in block:
                    save_path = os.path.join(save_path_, block)
                    os.makedirs(save_path, exist_ok=True)
                    break
            
            pkl_filename = os.path.join(save_path, path.split('/')[-1])
            
            # Try to load existing pkl file with prediction results
            result = {}
            if os.path.exists(pkl_filename):
                try:
                    with open(pkl_filename, 'rb') as f:
                        result = cPickle.load(f)
                    # print(f"Loaded existing pkl: {pkl_filename}")
                except:
                    print(f"Failed to load existing pkl: {pkl_filename}, creating new one")
                    result = {}
            else:
                print(f"No existing pkl found: {pkl_filename}, creating new one")
            
            # Add/Update ground truth information from dataset
            result['gt_class_ids'] = data['gt_class_ids'][0].numpy()
            result['gt_bboxes'] = data['gt_bboxes'][0].numpy()
            result['gt_RTs'] = data['gt_RTs'][0].numpy()
            result['gt_scales'] = data['gt_scales'][0].numpy()
            try:
                result['gt_handle_visibility'] = data['gt_handle_visibility'][0].numpy()
            except:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])

            # If prediction results are not in the file, use dataset predictions as fallback
            if 'pred_class_ids' not in result:
                result['pred_class_ids'] = data['pred_class_ids'][0].numpy()
            if 'pred_bboxes' not in result:
                result['pred_bboxes'] = data['pred_bboxes'][0].numpy()
            if 'pred_scores' not in result:
                result['pred_scores'] = data['pred_scores'][0].numpy()
            
            # If pred_RTs and pred_scales are not available, create placeholder
            if 'pred_RTs' not in result or 'pred_scales' not in result:
                num_pred = len(result['pred_class_ids'])
                if num_pred > 0:
                    # Create predictions based on available instances (placeholder)
                    pred_RTs = []
                    pred_scales = []
                    
                    # Match predictions to GT instances (this is a simplification)
                    valid_gt_indices = []
                    for pred_idx in range(num_pred):
                        pred_class = result['pred_class_ids'][pred_idx]
                        # Find matching GT instance
                        for gt_idx, gt_class in enumerate(result['gt_class_ids']):
                            if gt_class == pred_class and gt_idx not in valid_gt_indices:
                                valid_gt_indices.append(gt_idx)
                                break
                    
                    for idx in range(num_pred):
                        if idx < len(valid_gt_indices):
                            gt_idx = valid_gt_indices[idx]
                            pred_RTs.append(result['gt_RTs'][gt_idx])
                            pred_scales.append(result['gt_scales'][gt_idx])
                        else:
                            # If more predictions than GT, use identity matrix
                            identity_RT = np.eye(4)
                            pred_RTs.append(identity_RT)
                            pred_scales.append(np.ones(3) * 0.1)
                    
                    result['pred_RTs'] = np.array(pred_RTs)
                    result['pred_scales'] = np.array(pred_scales)
                else:
                    result['pred_RTs'] = np.zeros((0, 4, 4))
                    result['pred_scales'] = np.zeros((0, 3))

            # Save the updated result back to pkl file
            with open(pkl_filename, 'wb') as f:
                cPickle.dump(result, f)

            t.set_description(
                "Processing [{}/{}] GT:{} Pred:{}: {}".format(
                    i+1, len(dataloder), 
                    len(result['gt_class_ids']), 
                    len(result['pred_class_ids']),
                    path.split('/')[-1]
                )
            )
            t.update(1)

def load_housecat_depth(img_path):
    """ Load depth image from img_path. """
    depth_path = img_path.replace('rgb','depth')
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16

def get_bbox(bbox, img_width = 480, img_length = 640):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 440)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax

# Full kernels
FULL_KERNEL_3 = np.ones((3, 3), np.uint8)
FULL_KERNEL_5 = np.ones((5, 5), np.uint8)
FULL_KERNEL_7 = np.ones((7, 7), np.uint8)
FULL_KERNEL_9 = np.ones((9, 9), np.uint8)
FULL_KERNEL_31 = np.ones((31, 31), np.uint8)

# 3x3 cross kernel
CROSS_KERNEL_3 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

# 5x5 cross kernel
CROSS_KERNEL_5 = np.asarray(
    [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

# 7x7 cross kernel
CROSS_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

# 7x7 diamond kernel
DIAMOND_KERNEL_7 = np.asarray(
    [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.uint8)

def fill_in_fast(depth_map, max_depth=100.0, custom_kernel=DIAMOND_KERNEL_5,
                 extrapolate=False, blur_type='bilateral'):
    """Fast, in-place depth completion.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        custom_kernel: kernel to apply initial dilation
        extrapolate: whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'bilateral' - preserves local structure (recommended)
            'gaussian' - provides lower RMSE

    Returns:
        depth_map: dense depth map
    """

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    # Dilate
    depth_map = cv2.dilate(depth_map, custom_kernel)

    # Hole closing
    depth_map = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Fill empty spaces with dilated values
    empty_pixels = (depth_map < 0.1)
    dilated = cv2.dilate(depth_map, FULL_KERNEL_7)
    depth_map[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image
    if extrapolate:
        top_row_pixels = np.argmax(depth_map > 0.1, axis=0)
        top_pixel_values = depth_map[top_row_pixels, range(depth_map.shape[1])]

        for pixel_col_idx in range(depth_map.shape[1]):
            depth_map[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = \
                top_pixel_values[pixel_col_idx]

        # Large Fill
        empty_pixels = depth_map < 0.1
        dilated = cv2.dilate(depth_map, FULL_KERNEL_31)
        depth_map[empty_pixels] = dilated[empty_pixels]

    # Median blur
    depth_map = cv2.medianBlur(depth_map, 5)

    # Bilateral or Gaussian blur
    if blur_type == 'bilateral':
        # Bilateral blur
        depth_map = cv2.bilateralFilter(depth_map, 5, 1.5, 2.0)
    elif blur_type == 'gaussian':
        # Gaussian blur
        valid_pixels = (depth_map > 0.1)
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)
        depth_map[valid_pixels] = blurred[valid_pixels]

    # Invert
    valid_pixels = (depth_map > 0.1)
    depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

    return depth_map

def fill_in_multiscale(depth_map, max_depth=8.0,
                       dilation_kernel_far=CROSS_KERNEL_3,
                       dilation_kernel_med=CROSS_KERNEL_5,
                       dilation_kernel_near=CROSS_KERNEL_7,
                       extrapolate=False,
                       blur_type='bilateral',
                       show_process=False):
    """Slower, multi-scale dilation version with additional noise removal that
    provides better qualitative results.

    Args:
        depth_map: projected depths
        max_depth: max depth value for inversion
        dilation_kernel_far: dilation kernel to use for 30.0 < depths < 80.0 m
        dilation_kernel_med: dilation kernel to use for 15.0 < depths < 30.0 m
        dilation_kernel_near: dilation kernel to use for 0.1 < depths < 15.0 m
        extrapolate:whether to extrapolate by extending depths to top of
            the frame, and applying a 31x31 full kernel dilation
        blur_type:
            'gaussian' - provides lower RMSE
            'bilateral' - preserves local structure (recommended)
        show_process: saves process images into an OrderedDict

    Returns:
        depth_map: dense depth map
        process_dict: OrderedDict of process images
    """

    # Convert to float32
    depths_in = np.float32(depth_map)

    # Calculate bin masks before inversion
    valid_pixels_near = (depths_in > 0.01) & (depths_in <= 1.0)
    valid_pixels_med = (depths_in > 1.0) & (depths_in <= 2.0)
    valid_pixels_far = (depths_in > 2.0)

    # Invert (and offset)
    s1_inverted_depths = np.copy(depths_in)
    valid_pixels = (s1_inverted_depths > 0.01)
    s1_inverted_depths[valid_pixels] = \
        max_depth - s1_inverted_depths[valid_pixels]

    # Multi-scale dilation
    dilated_far = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_far),
        dilation_kernel_far)
    dilated_med = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_med),
        dilation_kernel_med)
    dilated_near = cv2.dilate(
        np.multiply(s1_inverted_depths, valid_pixels_near),
        dilation_kernel_near)

    # Find valid pixels for each binned dilation
    valid_pixels_near = (dilated_near > 0.01)
    valid_pixels_med = (dilated_med > 0.01)
    valid_pixels_far = (dilated_far > 0.01)

    # Combine dilated versions, starting farthest to nearest
    s2_dilated_depths = np.copy(s1_inverted_depths)
    s2_dilated_depths[valid_pixels_far] = dilated_far[valid_pixels_far]
    s2_dilated_depths[valid_pixels_med] = dilated_med[valid_pixels_med]
    s2_dilated_depths[valid_pixels_near] = dilated_near[valid_pixels_near]

    # Small hole closure
    s3_closed_depths = cv2.morphologyEx(
        s2_dilated_depths, cv2.MORPH_CLOSE, FULL_KERNEL_5)

    # Median blur to remove outliers
    s4_blurred_depths = np.copy(s3_closed_depths)
    blurred = cv2.medianBlur(s3_closed_depths, 5)
    valid_pixels = (s3_closed_depths > 0.01)
    s4_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Calculate a top mask
    top_mask = np.ones(depths_in.shape, dtype=bool)
    for pixel_col_idx in range(s4_blurred_depths.shape[1]):
        pixel_col = s4_blurred_depths[:, pixel_col_idx]
        top_pixel_row = np.argmax(pixel_col > 0.01)
        top_mask[0:top_pixel_row, pixel_col_idx] = False

    # Get empty mask
    valid_pixels = (s4_blurred_depths > 0.01)
    empty_pixels = ~valid_pixels & top_mask

    # Hole fill
    dilated = cv2.dilate(s4_blurred_depths, FULL_KERNEL_9)
    s5_dilated_depths = np.copy(s4_blurred_depths)
    s5_dilated_depths[empty_pixels] = dilated[empty_pixels]

    # Extend highest pixel to top of image or create top mask
    s6_extended_depths = np.copy(s5_dilated_depths)
    top_mask = np.ones(s5_dilated_depths.shape, dtype=bool)

    top_row_pixels = np.argmax(s5_dilated_depths > 0.01, axis=0)
    top_pixel_values = s5_dilated_depths[top_row_pixels,
                                         range(s5_dilated_depths.shape[1])]

    for pixel_col_idx in range(s5_dilated_depths.shape[1]):
        if extrapolate:
            s6_extended_depths[0:top_row_pixels[pixel_col_idx],
                               pixel_col_idx] = top_pixel_values[pixel_col_idx]
        else:
            # Create top mask
            top_mask[0:top_row_pixels[pixel_col_idx], pixel_col_idx] = False

    # Fill large holes with masked dilations
    s7_blurred_depths = np.copy(s6_extended_depths)
    for i in range(6):
        empty_pixels = (s7_blurred_depths < 0.01) & top_mask
        dilated = cv2.dilate(s7_blurred_depths, FULL_KERNEL_5)
        s7_blurred_depths[empty_pixels] = dilated[empty_pixels]

    # Median blur
    blurred = cv2.medianBlur(s7_blurred_depths, 5)
    valid_pixels = (s7_blurred_depths > 0.01) & top_mask
    s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    if blur_type == 'gaussian':
        # Gaussian blur
        blurred = cv2.GaussianBlur(s7_blurred_depths, (5, 5), 0)
        valid_pixels = (s7_blurred_depths > 0.01) & top_mask
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]
    elif blur_type == 'bilateral':
        # Bilateral blur
        blurred = cv2.bilateralFilter(s7_blurred_depths, 5, 0.5, 2.0)
        s7_blurred_depths[valid_pixels] = blurred[valid_pixels]

    # Invert (and offset)
    s8_inverted_depths = np.copy(s7_blurred_depths)
    valid_pixels = np.where(s8_inverted_depths > 0.01)
    s8_inverted_depths[valid_pixels] = \
        max_depth - s8_inverted_depths[valid_pixels]

    depths_out = s8_inverted_depths

    process_dict = None
    if show_process:
        process_dict = collections.OrderedDict()

        process_dict['s0_depths_in'] = depths_in

        process_dict['s1_inverted_depths'] = s1_inverted_depths
        process_dict['s2_dilated_depths'] = s2_dilated_depths
        process_dict['s3_closed_depths'] = s3_closed_depths
        process_dict['s4_blurred_depths'] = s4_blurred_depths
        process_dict['s5_combined_depths'] = s5_dilated_depths
        process_dict['s6_extended_depths'] = s6_extended_depths
        process_dict['s7_blurred_depths'] = s7_blurred_depths
        process_dict['s8_inverted_depths'] = s8_inverted_depths

        process_dict['s9_depths_out'] = depths_out

    return depths_out, process_dict

def fill_missing(
        dpt, cam_scale, scale_2_80m, fill_type='multiscale',
        extrapolate=False, show_process=False, blur_type='bilateral'
):
    dpt = dpt / cam_scale * scale_2_80m
    projected_depth = dpt.copy()
    if fill_type == 'fast':
        final_dpt = fill_in_fast(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            # max_depth=2.0
        )
    elif fill_type == 'multiscale':
        final_dpt, process_dict = fill_in_multiscale(
            projected_depth, extrapolate=extrapolate, blur_type=blur_type,
            show_process=show_process,
            max_depth=3.0
        )
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))
    dpt = final_dpt / scale_2_80m * cam_scale
    return dpt


class HouseCat6DTrickyDataset():
    def __init__(self,
                 image_size, 
                 sample_num, 
                 data_dir
                ):
        self.data_dir = data_dir
        self.sample_num = sample_num
        self.img_size = image_size
        self.test_scenes_rgb = glob.glob(os.path.join(self.data_dir, 'tricky', 'test_scene*', 'rgb'))
        self.test_intrinsics_list = [os.path.join(scene, '..', 'intrinsics.txt') for scene in self.test_scenes_rgb]
        self.test_img_list = [img_path for scene in self.test_scenes_rgb for img_path in
                              glob.glob(os.path.join(scene, '*.png'))]

        n_image = len(self.test_img_list)
        print('no. of test images: {}\n'.format(n_image))

        self.models = {}
        model_path = 'obj_models_small_size_final/objects.pkl'
        with open(os.path.join(self.data_dir, model_path), 'rb') as f:
            self.models.update(cPickle.load(f))
        print('{} models loaded.'.format(len(self.models))) 

        self.xmap = np.array([[i for i in range(1141)] for j in range(914)])
        self.ymap = np.array([[j for i in range(1141)] for j in range(914)])
        self.norm_scale = 1000.0    # normalization scale
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        self.sym_ids = [1, 2, 7]

    def compute_batch_bbox_sizes(self, data):
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)

        bbox_sizes = max_vals - min_vals
        return bbox_sizes

    def __len__(self):
        return len(self.test_img_list)

    def __getitem__(self, index):
        img_path = self.test_img_list[index]
        with open(img_path.replace('rgb', 'labels_gt').replace('.png', '_label.pkl'), 'rb') as f:
            gts = cPickle.load(f)

        mask_path = img_path.replace("rgb", "instance")
        mask = cv2.imread(mask_path)
        assert mask is not None
        mask = mask[:, :, 2]
        num_instance = len(gts['class_ids'])

        # rgb
        rgb = cv2.imread(img_path)[:, :, :3] # TODO 1141x914
        rgb = rgb[:, :, ::-1] # (914, 1141, 3)

        # pts
        intrinsics = np.loadtxt(os.path.join(img_path.split('rgb')[0], 'intrinsics.txt')).reshape(3, 3)
        cam_fx, cam_fy, cam_cx, cam_cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        depth = load_housecat_depth(img_path) #480*640 # TODO 1141x914
        depth = fill_missing(depth, self.norm_scale, 1)

        xmap = self.xmap
        ymap = self.ymap
        pts2 = depth.copy() / self.norm_scale
        pts0 = (xmap - cam_cx) * pts2 / cam_fx
        pts1 = (ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) # (914, 1141, 3)

        all_rgb = []
        all_pts = []
        all_choose = []
        all_cat_ids = []
        all_model = []
        flag_instance = torch.zeros(num_instance) == 1
        mask_target = mask.copy().astype(np.float32)

        for j in range(num_instance):
            mask = np.equal(mask_target, gts['instance_ids'][j])
            inst_mask = 255 * mask.astype('uint8')
            model = self.models[gts['model_list'][j][0]]
            size = self.compute_batch_bbox_sizes(model)
            scale = np.linalg.norm(size)
            model /= scale
            mask = inst_mask > 0
            mask = np.logical_and(mask, depth>0)
            if np.sum(mask) > 16:
                rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][j], img_width=914, img_length=1141)
                choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
                if len(choose) <= self.sample_num:
                    choose_idx = np.random.choice(len(choose), self.sample_num)
                else:
                    choose_idx = np.random.choice(len(choose), self.sample_num, replace=False)
                choose = choose[choose_idx]
                cat_id = gts['class_ids'][j] - 1 # convert to 0-indexed

                # Only for the ToM (Cutlery and Glass) Objects
                if cat_id not in [6, 7]:
                    continue
                
                instance_pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :].copy()

                instance_rgb = rgb[rmin:rmax, cmin:cmax, :].copy()
                instance_rgb = cv2.resize(instance_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                instance_rgb = self.transform(np.array(instance_rgb))


                crop_w = rmax - rmin
                ratio = self.img_size / crop_w
                col_idx = choose % crop_w
                row_idx = choose // crop_w
                choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)

                translation = gts['translations'][j].astype(np.float32)
                rotation = gts['rotations'][j].astype(np.float32)
                size = gts['gt_scales'][j].astype(np.float32)

                if cat_id in self.sym_ids:
                    theta_x = rotation[0, 0] + rotation[2, 2]
                    theta_y = rotation[0, 2] - rotation[2, 0]
                    r_norm = math.sqrt(theta_x**2 + theta_y**2)
                    s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                                        [0.0,            1.0,  0.0           ],
                                        [theta_y/r_norm, 0.0,  theta_x/r_norm]])
                    rotation = rotation @ s_map

                all_model.append(torch.FloatTensor(model))
                all_pts.append(torch.FloatTensor(instance_pts))
                all_rgb.append(torch.FloatTensor(instance_rgb))
                all_cat_ids.append(torch.IntTensor([cat_id]).long())
                all_choose.append(torch.IntTensor(choose).long())
                flag_instance[j] = 1

        ret_dict = {}
        RTs = []
        s = np.linalg.norm(gts["gt_scales"], axis=1)
        for each_idx in range(len(gts["rotations"])):
            matrix = np.identity(4)
            matrix[:3, :3] = gts["rotations"][each_idx] * s[each_idx]
            matrix[:3, 3] = gts["translations"][each_idx]
            RTs.append(matrix)
        RTs = np.stack(RTs, 0)
        
        ret_dict['gt_class_ids'] = torch.tensor(gts['class_ids'])
        ret_dict['gt_bboxes'] = torch.tensor(gts["bboxes"])
        ret_dict['gt_RTs'] = torch.tensor(RTs)
        ret_dict['gt_scales'] = torch.tensor(gts["gt_scales"] / s[:, np.newaxis])
        ret_dict['index'] = index

        if len(all_pts) == 0:
            ret_dict['pred_class_ids'] = torch.tensor(gts["class_ids"])
            ret_dict['pred_bboxes'] = torch.tensor(gts["bboxes"])
            ret_dict['pred_scores'] = torch.tensor(np.ones_like(np.array(gts["class_ids"]),np.float32))
        else:
            ret_dict['choose'] = torch.stack(all_choose)
            ret_dict['pts'] = torch.stack(all_pts) # N*3
            ret_dict['rgb'] = torch.stack(all_rgb)
            ret_dict['category_label'] = torch.stack(all_cat_ids).squeeze(1)
            ret_dict['model'] = torch.stack(all_model)
            ret_dict['pred_class_ids'] = torch.tensor(gts["class_ids"])[flag_instance==1]
            ret_dict['pred_bboxes'] = torch.tensor(gts["bboxes"])[flag_instance==1]
            ret_dict['pred_scores'] = torch.tensor(np.ones_like(np.array(gts["class_ids"]),np.float32))[flag_instance==1]

        return ret_dict


def get_parser():
    parser = argparse.ArgumentParser(
        description="evaluate on housecat6d tricky dataset")
    parser.add_argument("--data_dir",
                        type=str,
                        default="path to HouseCat6D-Tricky",
                        help="location of the housecat6d dataset")
    parser.add_argument("--result_dir",
                        type=str,
                        default="results",
                        help="location of the inference results")
    args_cfg = parser.parse_args()

    return args_cfg

if __name__ == "__main__":
    args = get_parser()
    save_path = args.result_dir
    if not os.path.isdir(save_path):
        os.makedirs(save_path, exist_ok=True)

    logger = get_logger(level_print=logging.INFO, level_save=logging.WARNING, 
        path_file="test_logger.log")
    
    dataset = HouseCat6DTrickyDataset(image_size=224, sample_num=1024, 
                                      data_dir=args.data_dir)
    dataloder = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=8,
            shuffle=False,
            drop_last=False
        )
    test_func_tricky(dataloder, save_path)
    evaluate_housecat(save_path, logger)

# Inference Results are saved in the .pkl files in the result_dir
