# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, num_classes, cost_class0: float = 1, cost_class1: float = 1,
                 curves_weight: float = 1, lower_weight: float = 1, upper_weight: float = 1, breakpoint_weight:float = 1):
        """Creates the matcher
        """

        super().__init__()
        self.cost_class0 = cost_class0 #3
        self.cost_class1 = cost_class1 #3
        threshold = 15 / 720.
        self.threshold = nn.Threshold(threshold**2, 0.)

        self.curves_weight = curves_weight #5
        self.lower_weight = lower_weight #2
        self.upper_weight = upper_weight #2
        self.breakpoint_weight = breakpoint_weight #2.5
        self.num_classes = num_classes

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"][:, :, :-1].flatten(0, 1).softmax(-1) #b*nq, 类别shu
        # tgt_ids  = torch.cat([tgt[:, 0] for tgt in targets]).long()
        out_prob0 = out_prob[..., :self.num_classes] #预测前段类别
        out_prob1 = out_prob[..., self.num_classes:] #预测后段类别
        tgt_ids0  = torch.cat([torch.div(tgt[:, 0],10) for tgt in targets]).long() #前段真值
        tgt_ids1  = torch.cat([tgt[:, 0]%10 for tgt in targets]).long() #后段真值
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class0 = -out_prob0[:, tgt_ids0] #b*nq 总车道数
        cost_class1 = -out_prob1[:, tgt_ids1] #b*np 总车道数

        out_exist_b = outputs["pred_logits"][:, :, -1].flatten(0, 1).sigmoid()
        out_bbox = outputs["pred_curves"] #b, nq, 系数个数
        tgt_uppers = torch.cat([tgt[:, 2] for tgt in targets]) #起点
        tgt_lowers = torch.cat([tgt[:, 1] for tgt in targets]) #终点
        tgt_breakpoints = torch.cat([tgt[:, -1] for tgt in targets]) #分割点

        tgt_exist_b = (tgt_lowers != tgt_breakpoints).long()
        cost_exist_b = (-out_exist_b[:, None] + tgt_exist_b.unsqueeze(0)).abs()

        # # Compute the L1 cost between lowers and uppers
        cost_lower = torch.cdist(out_bbox[:, :, 0].view((-1, 1)), tgt_lowers.unsqueeze(-1), p=1)
        cost_upper = torch.cdist(out_bbox[:, :, 1].view((-1, 1)), tgt_uppers.unsqueeze(-1), p=1)
        #cost_breakpoint = torch.cdist(out_bbox[:, :, -1].view((-1, 1)), tgt_breakpoints.unsqueeze(-1), p=1)

        # # Compute the poly cost
        tgt_points = torch.cat([tgt[:, 3:-1] for tgt in targets]) # 0~20 112
        tgt_xs = tgt_points[:, :tgt_points.shape[1] // 2]
        valid_xs = tgt_xs >= 0
        weights = (torch.sum(valid_xs, dtype=torch.float32) / torch.sum(valid_xs, dim=1, dtype=torch.float32))**0.5
        try:
            weights = weights / torch.max(weights)
        except:
            pass

        tgt_ys = tgt_points[:, tgt_points.shape[1] // 2:]
        out_polys = out_bbox[:, :, 2:-1].view((-1, 5))
        tgt_ys = tgt_ys.repeat(out_polys.shape[0], 1, 1)
        tgt_ys = tgt_ys.transpose(0, 2)
        tgt_ys = tgt_ys.transpose(0, 1)

        # Calculate the predicted xs
        out_xs = out_polys[:, 0] / (tgt_ys - out_polys[:, 1]) ** 2 + out_polys[:, 2] / (tgt_ys - out_polys[:, 1]) + \
                 out_polys[:, 3] + out_polys[:, 4] * tgt_ys - out_polys[:, 1]*out_polys[:, 4]
        tgt_xs = tgt_xs.repeat(out_polys.shape[0], 1, 1)
        tgt_xs = tgt_xs.transpose(0, 2)
        tgt_xs = tgt_xs.transpose(0, 1)

        cost_polys = torch.stack([torch.sum(torch.abs(tgt_x[valid_x] - out_x[valid_x]), dim=0) for tgt_x, out_x, valid_x in zip(tgt_xs, out_xs, valid_xs)], dim=-1)
        cost_polys = cost_polys * weights

        # # Final cost matrix
        C = self.cost_class0 * cost_class0 + self.cost_class1 * cost_class1 + self.curves_weight * cost_polys + \
            self.lower_weight * cost_lower + self.upper_weight * cost_upper + self.breakpoint_weight * cost_exist_b

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [tgt.shape[0] for tgt in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(num_classes, set_cost_class0, set_cost_class1,
                  curves_weight, lower_weight, upper_weight, breakpoint_weight):
    return HungarianMatcher(num_classes = num_classes, cost_class0=set_cost_class0, cost_class1=set_cost_class1,
                            curves_weight=curves_weight, lower_weight=lower_weight, upper_weight=upper_weight, breakpoint_weight=breakpoint_weight)
