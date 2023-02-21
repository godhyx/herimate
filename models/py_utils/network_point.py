import torch
import torch.nn as nn
from torchvision.ops import roi_align
from torch.nn.functional import interpolate
import math
import torch.nn.functional as F
import cv2
from config import system_configs
from .res18 import Res18
import time
from torchvision.ops import sigmoid_focal_loss
import numpy as np
from .KLD import jd_loss, xy_wh_r_2_xy_sigma
import torchvision.transforms.functional as Ftt
from scipy import ndimage


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU6(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Output_mask(nn.Module):
    def __init__(self, in_size, out_size):
        super(Output_mask, self).__init__()
        self.conv1 = Conv2D_BatchNorm_Relu(in_size, in_size//2, 1, 0, 1)
        self.conv2 = Conv2D_BatchNorm_Relu(in_size//2, in_size//4, 1, 0, 1)
        self.conv3 = Conv2D_BatchNorm_Relu(in_size//4, out_size, 1, 0, 1, acti = False)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class Conv2D_BatchNorm_Relu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=True, acti=True, dilation=1):
        super(Conv2D_BatchNorm_Relu, self).__init__()

        if acti:
            self.cbr_unit = nn.Sequential(nn.Conv2d(in_channels, n_filters, k_size,
                                                    padding=padding, stride=stride, bias=bias, dilation=dilation),
                                    nn.BatchNorm2d(n_filters),
                                    #nn.ReLU(inplace=True),)
                                    nn.PReLU(),)
        else:
            self.cbr_unit = nn.Conv2d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class Output(nn.Module):
    def __init__(self, in_size, out_size):
        super(Output, self).__init__()
        self.conv1 = Conv2D_BatchNorm_Relu(in_size, in_size//2, 3, 1, 1, dilation=1)
        self.conv2 = Conv2D_BatchNorm_Relu(in_size//2, in_size//4, 3, 1, 1, dilation=1)
        self.conv3 = Conv2D_BatchNorm_Relu(in_size//4, out_size, 1, 0, 1, acti = False)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class Conv1D_BatchNorm_Relu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, padding, stride, bias=True, acti=True, dilation=1):
        super(Conv1D_BatchNorm_Relu, self).__init__()

        if acti:
            self.cbr_unit = nn.Sequential(nn.Conv1d(in_channels, n_filters, k_size,
                                                    padding=padding, stride=stride, bias=bias, dilation=dilation),
                                    nn.BatchNorm1d(n_filters),
                                    #nn.ReLU(inplace=True),)
                                    nn.PReLU(),)
        else:
            self.cbr_unit = nn.Conv1d(in_channels, n_filters, k_size, padding=padding, stride=stride, bias=bias, dilation=dilation)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class Break_conv1d(nn.Module):
    def __init__(self, in_size, out_size):
        super(Break_conv1d, self).__init__()
        self.conv1 = Conv1D_BatchNorm_Relu(in_size, in_size//2, 5, 2, 2, dilation=1)
        self.conv2 = Conv1D_BatchNorm_Relu(in_size//2, in_size//4, 5, 2, 2, dilation=1)
        self.conv3 = Conv1D_BatchNorm_Relu(in_size//4, in_size//8, 5, 2, 2)
        self.conv4 = Conv1D_BatchNorm_Relu(in_size // 8, in_size // 16, 5, 2, 2)
        self.conv5 = Conv1D_BatchNorm_Relu(in_size // 16, out_size, 5, 0, 1, acti=False)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        return outputs


class Cls_conv1d(nn.Module):
    def __init__(self, in_size, out_size):
        super(Cls_conv1d, self).__init__()
        self.conv1 = Conv1D_BatchNorm_Relu(in_size, in_size // 2, 5, 2, 2, dilation=1)
        self.conv2 = Conv1D_BatchNorm_Relu(in_size // 2, in_size // 4, 5, 2, 2, dilation=1)
        self.conv3 = Conv1D_BatchNorm_Relu(in_size // 4, in_size // 8, 5, 2, 2)
        self.conv4 = Conv1D_BatchNorm_Relu(in_size // 8, in_size // 8, 5, 2, 2)
        self.conv5 = Conv1D_BatchNorm_Relu(in_size // 8, out_size, 3, 0, 2, acti=False)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        return outputs


class network_point(nn.Module):
    def __init__(self, hidden_dim=system_configs.attn_dim):
        super(network_point, self).__init__()
        hidden_dim = system_configs.attn_dim

        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=1, stride=1, padding=0,
                              bias=False)
        self.out_confidence = Output(hidden_dim * 2, 2)
        self.out_offset = Output(hidden_dim * 2, 2)
        #self.out_scope = Output(hidden_dim * 2, 1)
        #self.out_length = Output(hidden_dim * 2, 1)


        self.convlayer1 = BaseConv(hidden_dim * 2, hidden_dim * 2, 1, 1, act='relu')
        self.upsample1 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim * 2, 2, stride=2)
        self.out_mask = Output_mask(hidden_dim, 1)
        self.C3P4_1 = CSPLayer(hidden_dim * 2, hidden_dim, n=1, shortcut=True, depthwise=False, act="relu")

        self.Cls_conv1d = Output(hidden_dim * 2, system_configs.lane_categories)
        self.b_exist = Output(hidden_dim * 2, 1)
        #self.break_point = Output(hidden_dim * 2, 1)

    def forward(self, encoded_feature, imgC):
        encoded_feature, p40 = encoded_feature
        encoded_feature_points = self.conv1(encoded_feature)
        encoded_feature_cls = self.conv2(encoded_feature)

        out_mask = self.upsample1(encoded_feature_points)
        out_mask = self.C3P4_1(out_mask)
        out_mask = self.out_mask(out_mask)

        pooled_hs = roi_align(encoded_feature_points, imgC, output_size=1, spatial_scale=1 / 32, sampling_ratio=128)
        pooled_cls = roi_align(encoded_feature_cls, imgC, output_size=1, spatial_scale=1 / 32, sampling_ratio=128)

        out_confidence = self.out_confidence(pooled_hs)
        out_offset = self.out_offset(pooled_hs)

        out_cls = self.Cls_conv1d(pooled_cls) #n, num_cls, 1, 1
        out_b_exist = self.b_exist(pooled_cls) #n, 1, 1, 1

        result = [out_confidence, out_offset, out_mask, out_cls, out_b_exist]

        return result


class loss_point_onlyNeg(nn.Module):
    def __init__(self):
        super(loss_point_onlyNeg, self).__init__()
                                              #每个batch中gt的数量
    def get_neg_bbox(self, target_points, gt_nums_lane):
        """
        随机生成中心点，然后根据随机生成的中心点生成box，gt有180个采样点，生成的有80个采样点，通过记录每个batch，pre和gt
        采样点的有效个数，构建mask，通过mask计算对应pre与gt中心点的距离，通过mask，每一个pre对应180个gt，最小距离匹配，
        最小距离大于某个值的作为负样本
        """
        image_index_for_boxes = []
        gt_nums_lane_for_boxes = [[i] * gt_nums_lane[i] * system_configs.neg_sampling_rate for i in  #[[bath_id] * gt个数 * 采样点个数]
                                  range(len(gt_nums_lane))]
        for x in gt_nums_lane_for_boxes:
            for y in x:
                image_index_for_boxes.append(y) #将gt_nums_lane_for_boxes变成一个数组

        r = system_configs.roi_r
        centers = torch.rand(target_points.shape[0] * system_configs.neg_sampling_rate, 2) * torch.tensor([system_configs.input_w, system_configs.input_h]).cuda()
        boxes = torch.zeros((target_points.shape[0] * system_configs.neg_sampling_rate, 5))
        boxes[:, 0] = torch.Tensor(image_index_for_boxes)
        boxes[:, 1] = centers[:, 0] - r
        boxes[:, 2] = centers[:, 1] - r
        boxes[:, 3] = centers[:, 0] + r
        boxes[:, 4] = centers[:, 1] + r
        valid = (boxes[:, 1] > 0) & (boxes[:, 1] < system_configs.input_w) & (boxes[:, 2] >= 0) & (boxes[:, 2] < system_configs.input_h) & (
                    boxes[:, 3] >= 0) & (boxes[:, 3] < system_configs.input_w) & (boxes[:, 4] >= 0) & (boxes[:, 4] < system_configs.input_h)
        valid_boxes = boxes[valid]
        valid_centers = centers[valid]
        valid_index_per_img = [0 for i in range(1 + int(valid_boxes[:, 0].max().item()))]
        for i in valid_boxes[:, 0]:
            valid_index_per_img[int(i.item())] += 1 #每个batch有几个没超界的gt*80

        valid_index_per_img = valid_index_per_img
        new_valid_index_per_img = []
        old = 0
        for index in valid_index_per_img:
            new_valid_index_per_img.append((old, old + index)) #[(之前总个数，当前总个数-t-num*80)]
            old = old + index

        gt_points_per_img = target_points.split(gt_nums_lane)
        gt_points_per_img = [gt_points.view(-1, 2) for gt_points in gt_points_per_img]# [gt个数*180，2]
        gt_points_per_img = [x.shape[0] for x in gt_points_per_img]# [gt个数*180,...]

        new_gt_points_per_img = []
        old = 0
        for index in gt_points_per_img:
            new_gt_points_per_img.append((old, old + index))#[(之前总个数，当前总个数-t-num*180)]
            old = old + index

        target_points = target_points.view(-1, 2).unsqueeze(0).repeat(valid_centers.shape[0], 1, 1) #gt*80, gt*180, 2
        valid_centers = valid_centers.unsqueeze(1).repeat(1, target_points.shape[1], 1) #gt*80, gt*180, 2
        masks = torch.zeros(valid_centers.shape).bool() #gt*80, gt*180, 2
        log = []
        for i, j in zip(new_valid_index_per_img, new_gt_points_per_img):
            masks[i[0]:i[1], j[0]:j[1], :] = True
            log += [j[1] - j[0] for _ in range(i[1] - i[0])]#[j[1] - j[0] batch对应gt*180,重复gt*80次]
        dist = torch.norm((torch.tensor([system_configs.input_w, system_configs.input_h]).cuda() * target_points[masks[:, :, 0]]
                           - valid_centers[masks[:, :, 0]]), p=2, dim=-1) #gt*80 x gt*180
        dist = dist.split(log) #pre x tar展开, 通过mask使每一个pre刚好对应gt*180个真值,log重复了gt*80次，刚好取出一个batch
        min_dist = [x.min(-1)[0] for x in dist] #对每个box匹配
        selected = []
        for index, value in enumerate(min_dist):
            if 130 > value > (math.sqrt(2) * r):
                selected.append(index)

        imgC_Neg = torch.index_select(valid_boxes, 0, torch.Tensor(selected).int())
        del dist
        return imgC_Neg.cuda()

    def get_gt(self, tgt, imgC, valid_position, gt_idx):

        tgt = [tgt[tgt[:, 0] > 0] for tgt in tgt[1]]
        gt_nums_lane = [y.shape[0] for y in tgt]
        tgt = torch.stack([tgt[i][j] for i, j in zip(gt_idx[0], gt_idx[1])], 0)

        target_points = tgt[:, 3:-1]

        target_xs = target_points[:, :target_points.shape[1] // 2]
        target_ys = target_points[:, target_points.shape[1] // 2:]
        assert ((target_ys > -1) != (target_xs > -1)).sum().item() == 0
        points_num = torch.where(target_xs > -100, 1, 0).sum(-1)
        target_xs = target_xs[target_xs > -1]
        target_xs = target_xs.split(points_num.tolist())  # points for each lane
        target_ys = target_ys[target_ys > -1]
        target_ys = target_ys.split(points_num.tolist())  # points for each lane

        target_points = torch.ones((len(target_xs), 830, 2))
        target_points_sparse = torch.ones((len(target_xs), 180, 2))

        for index, line_x in enumerate(target_xs):
            line_x = line_x.unsqueeze(0).unsqueeze(0)
            line_x_dense = interpolate(line_x, size=830, mode='linear', align_corners=True)
            line_x_dense = line_x_dense[0, 0, :]
            target_points[index, :, 0] = line_x_dense[:]

            line_x_sparse = interpolate(line_x, size=180, mode='linear', align_corners=True)
            line_x_sparse = line_x_sparse[0, 0, :]
            target_points_sparse[index, :, 0] = line_x_sparse[:]

        for index, line_y in enumerate(target_ys):
            line_y = line_y.unsqueeze(0).unsqueeze(0)
            line_y_dense = interpolate(line_y, size=830, mode='linear', align_corners=True)
            line_y_dense = line_y_dense[0, 0, :]
            target_points[index, :, 1] = line_y_dense[:]

            line_y_sparse = interpolate(line_y, size=180, mode='linear', align_corners=True)
            line_y_sparse = line_y_sparse[0, 0, :]
            target_points_sparse[index, :, 1] = line_y_sparse[:]
        # target_points: for each GT lane, 830 interpolated points x 2 coord

        return target_points_sparse, gt_nums_lane

    def forward(self, imgC, xs, ys, valid_position, gt_idx, encoded_feature, Net):
        loss = 0
        target_points, gt_nums_lane = self.get_gt(ys, imgC, valid_position, gt_idx)

        imgC_Neg = self.get_neg_bbox(target_points, gt_nums_lane)

        point_out_Neg = Net(encoded_feature, imgC_Neg)
        pr_confidence_Neg = point_out_Neg[0][:, :, 0, 0]
        gt_confidence_Neg = torch.zeros(pr_confidence_Neg.shape[0]).cuda()
        loss_ce_Neg = F.cross_entropy(pr_confidence_Neg.unsqueeze(0).permute(0, 2, 1),
                                      gt_confidence_Neg.long().unsqueeze(-1).permute(1, 0))
        print('[POINT] loss_ce_Neg:', loss_ce_Neg.item())
        loss += loss_ce_Neg

        return loss, []


class loss_point(nn.Module):
    def __init__(self):
        super(loss_point, self).__init__()
        self.weight_dict = {'loss_p_ce': 1, 'loss_class': 1, 'loss_seg': 1, 'loss_breakpoints': 1, "loss_ce_focal": 1, "loss_offset": 1,"loss_smooth_offset": 1}

    def get_gt(self, tgt, imgC, valid_position, gt_idx):
        """
        首先将gt采样830个点，计算pre的box中心点 ，根据batch切分，gt同理，然后在每个batch中，计算中心点距离，对每个box选择距离最小的gt
        如果在gt中心点在box内则有效
        """
        # get gt_lanes_points
        tgt = [tgt[tgt[:, 0] > 0] for tgt in tgt[1]]
        gt_nums_lane = [y.shape[0] for y in tgt]

        # target_classes_f = torch.stack([tgt[i][j] for i, j in zip(gt_idx[0], gt_idx[1])], 0)[:, 0][posi_index]
        # target_classes_f = target_classes_f.long() // 10
        # target_classes_b = torch.stack([tgt[i][j] for i, j in zip(gt_idx[0], gt_idx[1])], 0)[:, 0][posi_index]
        # target_classes_b = target_classes_b.long() % 10
        #
        # target_breakpoints = torch.stack([tgt[i][j] for i, j in zip(gt_idx[0], gt_idx[1])], 0)[:, -1][posi_index]
        # target_lowers = torch.stack([tgt[i][j] for i, j in zip(gt_idx[0], gt_idx[1])], 0)[:, 1][posi_index]
        # breakpoint_t = target_breakpoints != target_lowers
        # target_breakpoints = (target_breakpoints - imgC80.reshape(-1, 80, 5)[:, 0, 2]) / system_configs.roi_r
        # target_breakpoints = target_breakpoints[breakpoint_t]



        tgt = torch.stack([tgt[i][j] for i, j in zip(gt_idx[0], gt_idx[1])], 0)
        target_points = tgt[:, 3:-1].cuda()
        target_classes_f = tgt[:, 0] // 10
        target_classes_b = tgt[:, 0] % 10
        target_breakpoint_index = target_classes_f != target_classes_b
        target_points_breakpoint_y = tgt[:, -1]
        target_xs = target_points[:, :target_points.shape[1] // 2]
        target_ys = target_points[:, target_points.shape[1] // 2:]
        assert ((target_ys > -1) != (target_xs > -1)).sum().item() == 0
        points_num = torch.where(target_xs > -100, 1, 0).sum(-1)
        target_xs = target_xs[target_xs > -1]
        target_xs = target_xs.split(points_num.tolist())  # points for each lane
        target_ys = target_ys[target_ys > -1]
        target_ys = target_ys.split(points_num.tolist())  # points for each lane
        target_points = torch.zeros((len(target_xs), 50, 4)).cuda()
        target_points[target_breakpoint_index, 0, 3] = target_points_breakpoint_y[target_breakpoint_index]
        target_points_sparse = torch.ones((len(target_xs), 30, 2)).cuda()
        for index, line_x in enumerate(target_xs):
            line_x = line_x.unsqueeze(0).unsqueeze(0)
            line_x_dense = interpolate(line_x, size=50, mode='linear', align_corners=True)
            line_x_dense = line_x_dense[0, 0, :]
            target_points[index, :, 0] = line_x_dense[:]
            line_x_sparse = interpolate(line_x, size=30, mode='linear', align_corners=True)
            line_x_sparse = line_x_sparse[0, 0, :]
            target_points_sparse[index, :, 0] = line_x_sparse[:]
        for index, line_y in enumerate(target_ys):
            line_y = line_y.unsqueeze(0).unsqueeze(0)
            line_y_dense = interpolate(line_y, size=50, mode='linear', align_corners=True)
            line_y_dense = line_y_dense[0, 0, :]
            target_points[index, :, 1] = line_y_dense[:]
            if target_points[index, 0, 3]:
                bigger_y = (line_y_dense > (target_points[index, 0, 3] + 5 / system_configs.input_h))
                smaller_y = (line_y_dense < (target_points[index, 0, 3] - 5 / system_configs.input_h))
                target_points[index, bigger_y, 3] = target_classes_f[index]
                target_points[index, smaller_y, 3] = target_classes_b[index]
                target_points[index, ~(smaller_y|bigger_y), 2] = 1
            else:
                target_points[index, :, 3] = target_classes_f[index]
            line_y_sparse = interpolate(line_y, size=30, mode='linear', align_corners=True)
            line_y_sparse = line_y_sparse[0, 0, :]
            target_points_sparse[index, :, 1] = line_y_sparse[:]

        gt_IDTmasks = self.get_gt_mask(target_points, gt_nums_lane)

        imgBox = imgC[:, 1:]  # for each predicted boxes, 4 coord
        imgCenter = torch.zeros(imgBox.shape[0], 2).cuda()
        imgCenter[:, 0] = imgBox[:, 0] / 2 + imgBox[:, 2] / 2
        imgCenter[:, 1] = imgBox[:, 1] / 2 + imgBox[:, 3] / 2
        imgCenter = imgCenter.unsqueeze(1).repeat(1, 50, 1)

        imgCln = imgC[:, 0]
        newimgCln = []

        for index in range(int(imgCln.max().item()) + 1):
            cnt = 0
            for l in imgCln.tolist():
                if l == index:
                    cnt += 1
            newimgCln.append(cnt)

        imgCenter_imgs = imgCenter[:, 0, :].split(newimgCln)

        newtgtln = []
        for index in range(int(gt_idx[0].max().item()) + 1):
            cnt = 0
            for l in gt_idx[0].tolist():
                if l == index:
                    cnt += 1
            newtgtln.append(cnt)
        target_points_imgs = target_points.split(newtgtln)

        r = system_configs.roi_r

        GT_confidence, GT_offset, GT_cls, GT_breakpoint = [], [], [], []

        for index, (imgCenter_img, target_points_img) in enumerate(zip(imgCenter_imgs, target_points_imgs)):
            target_points_img = target_points_img.view(-1, 4).unsqueeze(0).repeat(imgCenter_img.shape[0], 1, 1) * torch.tensor([system_configs.input_w, system_configs.input_h, 1, 1]).cuda()
            if target_points_img.shape[0] == 0:
                continue
            imgCenter_img = imgCenter_img.unsqueeze(1).repeat(1, target_points_img.shape[1], 1)
            dist = torch.norm(target_points_img[..., 0:2] - imgCenter_img, p=2, dim=-1)
            #min_dist_each_row = dist.min(-1)[0]
            min_indices = dist.min(-1)[1]
            min_indices = torch.clamp(min_indices, min=1, max=target_points_img.shape[1] - 3)
            min_indices_index = min_indices.t().unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 2)
            cls_indices = min_indices.t().unsqueeze(-1).unsqueeze(-1)
            min_points = torch.gather(target_points_img[..., 0:2], dim=1, index=min_indices_index) #80gt, 1, 2
            cls = torch.gather(target_points_img[..., 3].unsqueeze(-1), dim=1, index=cls_indices).squeeze(1)
            breakpoint = torch.gather(target_points_img[..., 2].unsqueeze(-1), dim=1, index=cls_indices).squeeze(1)
            gt_confidence = (torch.abs(min_points[:, 0, 0] - imgCenter_img[:, 0, 0]) < r) & (
                        torch.abs(min_points[:, 0, 1] - imgCenter_img[:, 0, 1]) < r)
            gt_offset = min_points.squeeze(1) #80gt, 2
            image_lt = imgCenter_img[:, 0, :] - r
            offset = (gt_offset - image_lt) / r / 2


            GT_confidence.append(gt_confidence)
            GT_offset.append(offset)
            GT_cls.append(cls)
            GT_breakpoint.append(breakpoint)
            # GT_angle.append(gt_angle)
            # GT_length.append(gt_length)
        del dist, cls_indices, min_indices_index
        GT_confidence = torch.cat(GT_confidence, 0)
        GT_offset = torch.cat(GT_offset, 0)
        GT_cls = torch.cat(GT_cls, 0)
        GT_breakpoint = torch.cat(GT_breakpoint, 0)
        # GT_angle = torch.cat(GT_angle, 0)
        # GT_length = torch.cat(GT_length, 0)

        return GT_confidence, GT_offset, target_points_sparse, gt_nums_lane, gt_IDTmasks, GT_cls.squeeze(-1), GT_breakpoint.squeeze(-1)#target_classes_f, target_classes_b, target_breakpoints, breakpoint_t  # TODO

    def get_gt_mask(self, gt_points, gt_nums_lane):

        gt_points = gt_points[:, :, 0:2] * torch.tensor([system_configs.input_w, system_configs.input_h]).cuda()
        gt_points[:, :, 0] = torch.clamp(gt_points[:, :, 0].floor(), 0, system_configs.input_w)
        gt_points[:, :, 1] = torch.clamp(gt_points[:, :, 1].floor(), 0, system_configs.input_h)
        gt_masks = torch.zeros(len(gt_nums_lane), system_configs.input_h, system_configs.input_w).cuda() #len(gt_nums_lane) batch个数
        gt_points = gt_points.split(gt_nums_lane)#根据每个batch的gt数量切分
        for index, img_gt_points in enumerate(gt_points):
            img_gt_points = img_gt_points.view(-1, 2)

            img_gt_points = torch.unique_consecutive(img_gt_points, dim=0).long() #去掉前后重复的一组点

            gt_masks[index, img_gt_points[:, 1].tolist(), img_gt_points[:, 0].tolist()] = True

        gt_masksc = gt_masks.cpu().numpy()
        gt_IDTmasks = torch.zeros(len(gt_nums_lane), system_configs.input_h // 16, system_configs.input_w // 16).cuda()
        for index, gt_mask in enumerate(gt_masksc):
            gt_IDTmask = 18 - np.clip(ndimage.distance_transform_edt(1 - gt_mask), 0, 18) #离车道线的点越近值越大
            gt_IDTmask = np.expand_dims(gt_IDTmask, -1)
            gt_IDTmask = cv2.resize(gt_IDTmask, (system_configs.input_w // 16, system_configs.input_h // 16))
            # inspect mask
            # if index == 0:
            #     cv2.imshow('a', gt_IDTmask)
            #     cv2.waitKey()
            gt_IDTmasks[index] = torch.Tensor(gt_IDTmask).cuda()

        return gt_IDTmasks

    def get_neg_bbox(self, target_points, gt_nums_lane):

        image_index_for_boxes = []
        gt_nums_lane_for_boxes = [[i] * gt_nums_lane[i] * system_configs.neg_sampling_rate for i in
                                  range(len(gt_nums_lane))]
        for x in gt_nums_lane_for_boxes:
            for y in x:
                image_index_for_boxes.append(y)

        r = system_configs.roi_r

        # centers = torch.rand(target_points.shape[0] * system_configs.neg_sampling_rate, 2).cuda() * 640
        # TODO only defect

        Centers = []
        splice = [len(a) // system_configs.neg_sampling_rate for a in gt_nums_lane_for_boxes]
        Target_points = target_points.split(splice)
        for target_point in Target_points:
            if target_point.shape[0] == 0:
                continue
            margin_x = torch.min((1 - target_point[:, :, 0].max()), target_point[:, :, 0].min()) * system_configs.input_w - 30
            centers_x = (torch.rand(target_point.shape[0] * system_configs.neg_sampling_rate, 1).cuda()) * (
                        system_configs.input_w - 2 * margin_x.item()) + margin_x.item()
            margin_y = torch.min((1 - target_point[:, :, 1].max()), target_point[:, :, 1].min()) * system_configs.input_h - 30
            centers_y = (torch.rand(target_point.shape[0] * system_configs.neg_sampling_rate, 1).cuda()) * (
                        system_configs.input_h - 2 * margin_y.item()) + margin_y.item()
            centers = torch.stack((centers_x, centers_y), -1)[:, 0, :]
            Centers.append(centers)
        centers = torch.cat(Centers, 0)

        boxes = torch.zeros((target_points.shape[0] * system_configs.neg_sampling_rate, 5)).cuda()
        boxes[:, 0] = torch.Tensor(image_index_for_boxes)
        boxes[:, 1] = centers[:, 0] - r
        boxes[:, 2] = centers[:, 1] - r
        boxes[:, 3] = centers[:, 0] + r
        boxes[:, 4] = centers[:, 1] + r
        valid = (boxes[:, 1] > 0) & (boxes[:, 1] < system_configs.input_w) & (boxes[:, 2] >= 0) & (boxes[:, 2] < system_configs.input_h) & (
                    boxes[:, 3] >= 0) & (boxes[:, 3] < system_configs.input_w) & (boxes[:, 4] >= 0) & (boxes[:, 4] < system_configs.input_h)
        valid_boxes = boxes[valid]
        valid_centers = centers[valid]
        valid_index_per_img = [0 for i in range(1 + int(valid_boxes[:, 0].max().item()))]
        for i in valid_boxes[:, 0]:
            valid_index_per_img[int(i.item())] += 1

        valid_centers_per_img = valid_centers.split(valid_index_per_img)
        boxes_per_img = valid_boxes.split(valid_index_per_img)
        gt_points_per_img = target_points.split(gt_nums_lane)
        gt_points_per_img = [gt_points.view(-1, 2) for gt_points in gt_points_per_img]

        imgC_Neg = []
        for centers, tgts, boxes in zip(valid_centers_per_img, gt_points_per_img, boxes_per_img):
            if tgts.shape[0] == 0:
                continue
            centers = centers.cuda().unsqueeze(1).repeat(1, tgts.shape[0], 1) #num_gt*80, num_gt*180, 2
            tgts = tgts.cuda().unsqueeze(0).repeat(centers.shape[0], 1, 1) * torch.tensor([system_configs.input_w, system_configs.input_h]).cuda() #num_gt*80, num_gt*180, 2
            dist = torch.norm(centers - tgts, p=2, dim=-1)
            min_dist_each_row = dist.min(-1)[0] > (math.sqrt(2) * r + 1) #在box外
            neg_box_per_img = boxes[min_dist_each_row]
            imgC_Neg.append(neg_box_per_img)
        imgC_Neg = torch.cat(imgC_Neg, 0)

        return imgC_Neg.cuda()

    def forward(self, point_out, imgC, xs, ys, valid_position, gt_idx, encoded_feature, Net):
        losses = {}
        pr_confidence, pr_offset, pr_mask, pr_cls, pr_b_exist = point_out
        pr_confidence, pr_offset, pr_mask , pr_cls, pr_b_exist = pr_confidence[:, :, 0, 0], \
                                                                 pr_offset[:, :, 0, 0], \
                                                                 pr_mask[:, 0, :, :], \
                                                                 pr_cls[:, :, 0, 0], \
                                                                 pr_b_exist[:, :, 0, 0].squeeze(-1)
        gt_confidence, gt_offset, target_points, gt_nums_lane, gt_IDTmasks, gt_cls, gt_breakpoint = self.get_gt(ys, imgC, valid_position, gt_idx)

        cur = 0
        cnt = 0
        splice = []
        for i in valid_position[0]:
            if i == cur:
                cnt += 1
            else:
                splice.append(cnt)
                cnt = 1
                cur += 1
        splice.append(cnt) #每个gt有多少个有效采样点

        imgC_Neg = self.get_neg_bbox(target_points, gt_nums_lane)
        loss_seg = F.l1_loss(pr_mask, gt_IDTmasks) * 0.6
        #losses["loss"] += loss_seg
        losses["loss_seg"] = loss_seg
        #print("[POINT]loss_seg:" + str(loss_seg.item()))

        point_out_Neg = Net(encoded_feature, imgC_Neg)
        pr_confidence_Neg = point_out_Neg[0][:, :, 0, 0]
        gt_confidence_Neg = torch.zeros(pr_confidence_Neg.shape[0]).cuda()
        loss_ce = 1 * F.cross_entropy(pr_confidence.unsqueeze(0).permute(0, 2, 1),
                                      gt_confidence.cuda().long().unsqueeze(-1).permute(1, 0))
        pr_confidence_focal = torch.cat((pr_confidence, pr_confidence_Neg))
        gt_confidence_focal = torch.cat((gt_confidence, gt_confidence_Neg))
        loss_ce_focal = 25 * sigmoid_focal_loss(torch.max(pr_confidence_focal, -1)[0],
                                                gt_confidence_focal.float().cuda(), alpha=0.8, reduction='mean')

        losses["loss_p_ce"] = loss_ce
        losses["loss_ce_focal"] = loss_ce_focal

        if gt_confidence.sum().item():
            loss_offset = 40 * F.mse_loss(gt_offset[gt_confidence].cuda(), pr_offset[gt_confidence])
            # loss_theta = 1 * F.l1_loss(gt_theta[gt_confidence].cuda(), pr_theta[gt_confidence])
            # loss_length =  F.l1_loss(gt_length[gt_confidence].cuda() / system_configs.roi_r, pr_length[gt_confidence])
            #losses["loss"] += loss_offset
            losses["loss_offset"] = loss_offset
            # loss += loss_theta
            # loss += loss_length
            loss_breakpoints = sigmoid_focal_loss(pr_b_exist[gt_confidence], gt_breakpoint[gt_confidence], alpha=0.99, reduction='sum')
            losses["loss_breakpoints"] = loss_breakpoints

            cls_index = gt_cls[gt_confidence] != 0
            loss_class = F.cross_entropy(pr_cls[gt_confidence][cls_index], gt_cls[gt_confidence][cls_index].long())
            losses["loss_class"] = loss_class
            del cls_index
            # xywhr_pr = torch.stack((pr_offset[gt_confidence][:, 0], pr_offset[gt_confidence][:, 1], pr_length[gt_confidence], pr_length[gt_confidence]/4, pr_theta[gt_confidence]), -1)
            # xywhr_gt = torch.stack((gt_offset[gt_confidence][:, 0], gt_offset[gt_confidence][:, 1], gt_length[gt_confidence], gt_length[gt_confidence]/4, gt_theta[gt_confidence]), -1)
            # GT = xy_wh_r_2_xy_sigma(xywhr_gt)
            # PR = xy_wh_r_2_xy_sigma(xywhr_pr)
            # loss_kld = 0.5 * torch.mean(jd_loss(PR, GT))
            #print("[POINT]loss_offset:" + str(loss_offset.item()) + "    loss_theta:" + str(loss_theta.item()) + "    loss_length:" + str(loss_length.item()) + "    loss_kld:" + str(loss_kld.item()))
            # loss += loss_kld

            #print("[POINT]loss_offset:"+ str(loss_offset.item()))

            imgC_splice = imgC.split(splice)
            gt_confidence_splice = gt_confidence.split(splice)
            pr_offset_splice = pr_offset.split(splice)
            # pr_theta_splice = pr_theta.split(splice)

            # loss_smooth_theta = []
            loss_smooth_offset = []

            for imgC_splice_lane, gt_confidence_splice_lane, pr_offset_splice_lane in zip(
                    imgC_splice,
                    gt_confidence_splice,
                    pr_offset_splice,
                    # pr_theta_splice
            ):
                if gt_confidence_splice_lane.sum().item() > 1:
                    pr_pos_xy = (imgC_splice_lane[:, 1:3] + system_configs.roi_r * 2 * pr_offset_splice_lane)[
                        gt_confidence_splice_lane]
                    dist_adjasant_point = torch.norm(pr_pos_xy[1:, :] - pr_pos_xy[:-1, :], 2, -1)
                    dist_weight = dist_adjasant_point / (system_configs.roi_r * 2)

                    # pr_theta_splice_lane = pr_theta_splice_lane[gt_confidence_splice_lane]
                    # dist_theta_point_change = (pr_theta_splice_lane[1:] - pr_theta_splice_lane[:-1]) / (dist_weight + 1e-12)
                    # dist_theta_point_change_change = abs(dist_theta_point_change[1:] - dist_theta_point_change[:-1])
                    # if dist_theta_point_change_change.shape[0]:
                    #     dist_theta_point_loss = dist_theta_point_change_change.mean() / 5
                    #     loss_smooth_theta.append(dist_theta_point_loss)

                    dist_point_change = (pr_pos_xy[1:, :] - pr_pos_xy[:-1, :]) / (
                            dist_weight.unsqueeze(-1).repeat(1, 2) + 1e-12)
                    dist_point_change_change = abs(dist_point_change[1:, :] - dist_point_change[:-1, :])
                    if dist_point_change_change.shape[0]:
                        dist_point_loss = dist_point_change_change.mean() / 5
                        loss_smooth_offset.append(dist_point_loss)

                    dist_point_change = (pr_pos_xy[1:, :] - pr_pos_xy[:-1, :]) / (
                            dist_weight.unsqueeze(-1).repeat(1, 2) + 1e-12)
                    dist_point_change_change = abs(dist_point_change[1:, :] - dist_point_change[:-1, :])
                    if dist_point_change_change.shape[0]:
                        dist_point_loss = dist_point_change_change.mean() / 5
                        loss_smooth_offset.append(dist_point_loss)

            loss_smooth_offset = sum(loss_smooth_offset) / (len(loss_smooth_offset) + 1e-8)
            if isinstance(loss_smooth_offset, int) or isinstance(loss_smooth_offset, float):
                losses["loss_smooth_offset"] = loss_ce.new_zeros(loss_ce.shape)
            else:
                losses["loss_smooth_offset"] = loss_smooth_offset
                #print("    loss_smooth_offset:" + str(loss_smooth_offset.item()))

        else:
            losses["loss_smooth_offset"] = loss_ce.new_zeros(loss_ce.shape)
            #print('all sampled imgC are Neg')

        loss_dict_reduced_scaled = {f'{k}': v * self.weight_dict[k] for k, v in losses.items() if k in self.weight_dict}
        loss = sum(loss_dict_reduced_scaled[k] for k in loss_dict_reduced_scaled.keys())


        return (loss, loss_dict_reduced_scaled), gt_confidence.cuda().sum()