# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torchvision
import cv2

RED = (0, 0, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (115, 181, 34)
BLUE = (255, 0, 0)
CYAN = (255, 128, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
PINK   = (180, 105, 255)

# SBC_colors = [ORANGE, RED, CYAN, DARK_GREEN, GREEN, BLUE, YELLOW, PURPLE, PINK]
SBC_colors = [ORANGE, ORANGE, ORANGE, RED, RED, RED, CYAN, CYAN, CYAN]
COLOR = [RED, GREEN, DARK_GREEN, BLUE, CYAN, YELLOW, ORANGE, PURPLE, PINK]

KPS_colors = [DARK_GREEN, DARK_GREEN, YELLOW, YELLOW, PINK]

def save_batch_image_with_curves(batch_image,
                                batch_curves,
                                batch_labels,
                                file_name,
                                nrow=4,
                                padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    # print(file_name)
    B, C, H, W = batch_image.size()
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            boxes = batch_curves[k]
            labels = batch_labels[k]
            num_box = boxes.shape[0]
            i = 0
            for n in range(num_box):
                lane = boxes[:, 3:][n]
                xs = lane[:len(lane) // 2]
                ys = lane[len(lane) // 2:-1]
                breakpoints = lane[-1] *H
                ys = ys[xs >= 0] * H
                xs = xs[xs >= 0] * W
                cls = labels[n]
                '''
                if (cls > 0 and cls < 10):
                    for jj, xcoord, ycoord in zip(range(xs.shape[0]), xs, ys):
                        j_x = x * width + padding + xcoord
                        j_y = y * height + padding + ycoord
                        cv2.circle(ndarr, (int(j_x), int(j_y)), 2, PINK, 10)
                    i += 1
                '''
                cls0 = cls//10
                cls1 = cls%10
                if cls0 == 0 or cls1 == 0:
                    continue
                for jj, xcoord, ycoord in zip(range(xs.shape[0]), xs, ys):
                    j_x = x * width + padding + xcoord
                    j_y = y * height + padding + ycoord
                    # breakpoint = y * height + padding + ycoord
                    if ycoord>=breakpoints:
                        color = COLOR[cls0-1]
                    else:
                        color = COLOR[cls1-1]
                    cv2.circle(ndarr, (int(j_x), int(j_y)), 2, color, 5)
                i += 1
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_batch_image_with_dbs(batch_image,
                              batch_curves,
                              batch_labels,
                              file_name,
                              nrow=4,
                              padding=2,
                              breakpoint_prob=0.7):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    # print(file_name)
    B, C, H, W = batch_image.size()
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            pred = batch_curves[k].cpu().numpy()  # 10 7
            labels = batch_labels[k].cpu().numpy()  # 10
            #print(labels)
            #pred = pred[labels == 1] # only draw lanes
            num_pred = pred.shape[0]
            if num_pred > 0:
                for n, lane in enumerate(pred):
                    existe_b = lane[0]
                    lane = lane[1:]
                    lower, upper = lane[0], lane[1]
                    breakpoint = lane[-1]
                    lane = lane[2:-1]
                    ys = np.linspace(lower, upper, num=100)
                    points = np.zeros((len(ys), 2), dtype=np.int32)
                    points[:, 1] = (ys * H).astype(int)
                    # Calculate the predicted xs
                    points[:, 0] = ((lane[0] / (ys - lane[1]) ** 2 + lane[2] / (ys - lane[1]) + lane[3] + lane[4] * ys - lane[1]*lane[4])
                                    * W).astype(int)
                    points = points[(points[:, 0] > 0) & (points[:, 0] < W)]
                    points[:, 0] += x * width + padding
                    points[:, 1] += y * height + padding

                    BP_y = (breakpoint * H).astype(int)
                    BP_x =  ((lane[0] / (breakpoint - lane[1]) ** 2 + lane[2] / (breakpoint - lane[1]) + lane[3] + lane[4] * breakpoint - lane[1]*lane[4])
                                    * W).astype(int)
                    BP_x += x * width + padding
                    BP_y += y * height + padding

                    cls = labels[n]
                    #print("+++++++",cls)
                    cls0 = cls//10
                    cls1 = cls%10
                    if cls0 == 0 or cls1 == 0:
                        continue
                    if existe_b < breakpoint_prob:
                        BP_y = (lower * H).astype(int) + y * height + padding
                    for current_point, next_point in zip(points[:-1], points[1:]):
                        if current_point[1]>=BP_y:
                            color = COLOR[cls0-1]
                        else:
                            color = COLOR[cls1-1]
                        cv2.line(ndarr, tuple(current_point), tuple(next_point), color, thickness=2)
                        # cv2.circle(ndarr, (BP_x,BP_y), 2,(0,255,255),5)
            k = k + 1
    cv2.imwrite(file_name, ndarr)

def save_debug_images_boxes(input, tgt_curves, tgt_labels,
                            pred_curves, pred_labels, prefix=None):
    save_batch_image_with_curves(
        input, tgt_curves, tgt_labels,
        '{}_gt.jpg'.format(prefix)
    )

    save_batch_image_with_dbs(
        input, pred_curves, pred_labels,
        '{}_pred.jpg'.format(prefix)
    )

