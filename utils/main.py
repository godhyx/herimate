import torch
import numpy as np


window_size = [7, 7]
num_heads = 8
shift_size = 4
H = 32
W = 32

logit_scale = torch.nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
relative_coords_h = torch.arange(-(window_size[0] - 1), window_size[0], dtype=torch.float32)
relative_coords_w = torch.arange(-(window_size[1] - 1), window_size[1], dtype=torch.float32)
relative_coords_table = torch.meshgrid([relative_coords_h, relative_coords_w])
relative_coords_table = torch.stack(relative_coords_table)
relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)
relative_coords_table[:, :, :, 0] /= (window_size[0] - 1)
relative_coords_table[:, :, :, 1] /= (window_size[1] - 1)
relative_coords_table *= 8
a = torch.log2(torch.abs(relative_coords_table)+1)
b = torch.sign(relative_coords_table)
c =  b * a
relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)


coords_h = torch.arange(window_size[0])
coords_w = torch.arange(window_size[1])
coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
relative_coords[:, :, 1] += window_size[1] - 1
relative_coords[:, :, 0] *= 2 * window_size[1] - 1
relative_position_index = relative_coords.sum(-1)

h_slices = (slice(0, -7),
            slice(-7, -shift_size),
            slice(-shift_size, None))
w_slices = (slice(0, -7),
            slice(-7, -shift_size),
            slice(-shift_size, None))
cnt = 0
img_mask = torch.zeros((1, H, W, 1)) + 5
for h in h_slices:
    for w in w_slices:
        img_mask[:, h, w, :] = cnt
        cnt += 1
print(relative_coords_table)