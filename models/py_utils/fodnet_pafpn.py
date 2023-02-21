#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
import torch.onnx as ONNX

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class FODNetPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
        res_dims=None,
        hidden_dim=None
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.input_proj = nn.Conv2d(res_dims, hidden_dim, kernel_size=1)

    def to_onnx(self,file_name):
        self.eval()
        x = torch.randn(1,3,640,960)
        x = x.cuda()
        with torch.no_grad():
            ONNX.export(self, 
                         x,
                         file_name,
                         opset_version=9,
                         do_constant_folding=True,  
                         verbose = True,
                         input_names=["input"], 
                         output_names=["output"])

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        out_features = self.backbone(input)#darknet2,3,4,5
        # print(out_features.shape)
        features = [out_features[f] for f in self.in_features]#darknet3,4,5
        
        [x2, x1, x0] = features#32,32,24,120 32,64,12,60 32,128,6,30
        fpn_out0 = self.lateral_conv0(x0)#B,64,6,30  
        # f_out0 = self.upsample1(fpn_out0)  
        f_out0 = self.upsample(fpn_out0)#B,64,12,60 
        # print(fpn_out0.shape, f_out0.shape)
        f_out0 = torch.cat([f_out0, x1], 1)#B,128,12,60   
        f_out0 = self.C3_p4(f_out0) #B,64,12,60 

        fpn_out1 = self.reduce_conv1(f_out0) #32,32,12,60 
        # f_out1 = self.upsample2(fpn_out1)  
        f_out1 = self.upsample(fpn_out1) #32,32,24,120
        # print(fpn_out1.shape, f_out1.shape)
        f_out1 = torch.cat([f_out1, x2], 1)#32,64,24,120  
        pan_out2 = self.C3_p3(f_out1) #32,32,24,120 
        p_out1 = self.bu_conv2(pan_out2)  
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  
        pan_out1 = self.C3_n3(p_out1)  
        p_out0 = self.bu_conv1(pan_out1)  
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  
        pan_out0 = self.C3_n4(p_out0)  
        p = self.input_proj(pan_out0)

        # outputs = (pan_out2, pan_out1, pan_out0)
        # return outputs
        return p
