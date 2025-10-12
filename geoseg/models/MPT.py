import math
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from typing import Optional

from mmengine.dist import get_dist_info
from mmseg.models.decode_heads import UPerHead
from .D2LORA import D2LoRALinear

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)

class DepthWiseConv2d(nn.Module):
    def __init__(self, in_ch, out_ch,kernel_size,stride,padding,bias=True):
        super(DepthWiseConv2d, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    bias=bias,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    bias=bias)
    def forward(self, input):
        return self.point_conv(self.depth_conv(input))

class DepConvBNReLU2d(nn.Module):
    def __init__(self, in_channels, out_channels,k=3,s=1,p=1,res=False):
        super(DepConvBNReLU2d, self).__init__()
        self.res = res
        self.conv1 = DepthWiseConv2d(in_channels, out_channels, kernel_size=k, padding=p ,stride=s, bias=False)
        self.BN = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        if self.res:
            res = x
        x = self.conv1(x)
        x = self.BN(x)
        if self.res:
            return self.relu(x)+res
        else:
            return self.relu(x)

class DepConvBNGELU2d(nn.Module):
    def __init__(self, in_channels, out_channels,k=3,s=1,p=1,res=False):
        super(DepConvBNGELU2d, self).__init__()
        self.res = res
        self.conv1 = DepthWiseConv2d(in_channels, out_channels, kernel_size=k, padding=p ,stride=s, bias=False)
        self.BN = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
    def forward(self, x):
        if self.res:
            res = x
        x = self.conv1(x)
        x = self.BN(x)
        if self.res:
            return self.gelu(x)+res
        else:
            return self.gelu(x)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.window_size = window_size
        q_size = window_size[0]
        kv_size = q_size
        rel_sp_dim = 2 * q_size - 1
        self.full_attn_rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))  # 2ws-1,C'
        self.full_attn_rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))  # 2ws-1,C'
        #####################################################################
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)



    def forward(self, x, H, W, rel_pos_bias=None):
        B, N, C = x.shape
        # qkv_bias = None
        # if self.q_bias is not None:
        #     qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 3，B，H，N，C
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) # B，H，N，C

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B,H,N,N

        attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.full_attn_rel_pos_h,
                                    self.full_attn_rel_pos_w)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def calc_rel_pos_spatial(
        attn,
        q,
        q_shape,
        k_shape,
        rel_pos_h,
        rel_pos_w,
):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
            torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )

    # dist_h [1-ws,ws-1]->[0,2ws-2]

    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
            torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    # get pos encode, qwh, kwh, C'

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)  # B, H, qwh, qww, C
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)  # B, H, qwh, qww, C'; qwh, kWh, C' -> B,H,qwh,qww,kwh
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)  # B,H,qwh,qww,kww

    # attn: B,H,qwh,qww,kwh,kww

    attn[:, :, sp_idx:, sp_idx:] = (
            attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, :, None]
            + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class RotatedVariedSizeWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, out_dim=None, window_size=1, qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0, attn_head_dim=None, relative_pos_embedding=True, learnable=True,
                 restart_regression=True,
                 attn_window_size=None, shift_size=0, img_size=(1, 1), num_deform=None):
        super().__init__()

        window_size = window_size[0]

        self.img_size = to_2tuple(img_size)
        self.num_heads = num_heads
        self.dim = dim
        out_dim = out_dim or dim
        self.out_dim = out_dim
        self.relative_pos_embedding = relative_pos_embedding
        head_dim = dim // self.num_heads
        self.ws = window_size
        attn_window_size = attn_window_size or window_size
        self.attn_ws = attn_window_size or self.ws

        q_size = window_size
        rel_sp_dim = 2 * q_size - 1
        self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.learnable = learnable
        self.restart_regression = restart_regression
        if self.learnable:
            # if num_deform is None, we set num_deform to num_heads as default

            if num_deform is None:
                num_deform = 1
            self.num_deform = num_deform

            self.sampling_offsets = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size, stride=window_size),
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads * self.num_deform * 2, kernel_size=1, stride=1)
            )
            self.sampling_scales = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size, stride=window_size),
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads * self.num_deform * 2, kernel_size=1, stride=1)
            )
            # add angle
            self.sampling_angles = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size, stride=window_size),
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads * self.num_deform * 1, kernel_size=1, stride=1)
            )

        self.shift_size = shift_size % self.ws
        # self.left_size = self.img_size
        #        if min(self.img_size) <= self.ws:
        #            self.shift_size = 0

        # if self.shift_size > 0:
        #     self.padding_bottom = (self.ws - self.shift_size + self.padding_bottom) % self.ws
        #     self.padding_right = (self.ws - self.shift_size + self.padding_right) % self.ws

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, out_dim * 3, bias=qkv_bias)

        # self.qkv = nn.Conv2d(dim, out_dim * 3, 1, bias=qkv_bias)
        # self.kv = nn.Conv2d(dim, dim*2, 1, bias=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        ######################################################################
        self.proj_drop = nn.Dropout(proj_drop)

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((window_size + attn_window_size - 1) * (window_size + attn_window_size - 1),
                            num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.attn_ws)
            coords_w = torch.arange(self.attn_ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.attn_ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.attn_ws - 1
            relative_coords[:, :, 0] *= 2 * self.attn_ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)
            print('The relative_pos_embedding is used')

    def forward(self, x, H, W):

        B, N, C = x.shape
        assert N == H * W
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        b, _, h, w = x.shape
        shortcut = x
        # assert h == self.img_size[0]
        # assert w == self.img_size[1]
        # if self.shift_size > 0:
        padding_td = (self.ws - h % self.ws) % self.ws
        padding_lr = (self.ws - w % self.ws) % self.ws
        padding_top = padding_td // 2
        padding_down = padding_td - padding_top
        padding_left = padding_lr // 2
        padding_right = padding_lr - padding_left

        # padding on left-right-up-down
        expand_h, expand_w = h + padding_top + padding_down, w + padding_left + padding_right

        # window num in padding features
        window_num_h = expand_h // self.ws
        window_num_w = expand_w // self.ws

        image_reference_h = torch.linspace(-1, 1, expand_h).to(x.device)
        image_reference_w = torch.linspace(-1, 1, expand_w).to(x.device)
        image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2,
                                                                                                       1).unsqueeze(
            0)  # 1, 2, H, W

        # position of the window relative to the image center
        window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=self.ws)  # 1,2, nh, nw
        image_reference = image_reference.reshape(1, 2, window_num_h, self.ws, window_num_w,
                                                  self.ws)  # 1, 2, nh, ws, nw, ws
        assert window_num_h == window_reference.shape[-2]
        assert window_num_w == window_reference.shape[-1]

        window_reference = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)  # 1,2, nh,1, nw,1

        # coords of pixels in each window

        base_coords_h = torch.arange(self.attn_ws).to(x.device) * 2 * self.ws / self.attn_ws / (expand_h - 1)  # ws
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(self.attn_ws).to(x.device) * 2 * self.ws / self.attn_ws / (expand_w - 1)
        base_coords_w = (base_coords_w - base_coords_w.mean())
        # base_coords = torch.stack(torch.meshgrid(base_coords_w, base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, 1, self.attn_ws, 1, self.attn_ws)

        # extend to each window
        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)  # ws -> 1,ws -> nh,ws
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == self.attn_ws
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)  # nw,ws
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == self.attn_ws
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)  # nh*ws
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)  # nw*ws

        window_coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2,
                                                                                                               1).reshape(
            1, 2, window_num_h, self.attn_ws, window_num_w, self.attn_ws)  # 1, 2, nh, ws, nw, ws
        # base_coords = window_reference+window_coords
        base_coords = image_reference

        # padding feature
        x = torch.nn.functional.pad(x, (padding_left, padding_right, padding_top, padding_down))

        if self.restart_regression:
            # compute for each head in each batch
            coords = base_coords.repeat(b * self.num_heads, 1, 1, 1, 1, 1)  # B*nH, 2, nh, ws, nw, ws
        if self.learnable:
            # offset factors
            sampling_offsets = self.sampling_offsets(x)

            num_predict_total = b * self.num_heads * self.num_deform

            sampling_offsets = sampling_offsets.reshape(num_predict_total, 2, window_num_h, window_num_w)
            sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (h // self.ws)
            sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (w // self.ws)

            # scale fators
            sampling_scales = self.sampling_scales(x)  # B, heads*2, h // window_size, w // window_size
            sampling_scales = sampling_scales.reshape(num_predict_total, 2, window_num_h, window_num_w)

            # rotate factor
            sampling_angle = self.sampling_angles(x)
            sampling_angle = sampling_angle.reshape(num_predict_total, 1, window_num_h, window_num_w)

            # first scale

            window_coords = window_coords * (sampling_scales[:, :, :, None, :, None] + 1)

            # then rotate around window center

            window_coords_r = window_coords.clone()

            # 0:x,column, 1:y,row

            window_coords_r[:, 0, :, :, :, :] = -window_coords[:, 1, :, :, :, :] * torch.sin(
                sampling_angle[:, 0, :, None, :, None]) + window_coords[:, 0, :, :, :, :] * torch.cos(
                sampling_angle[:, 0, :, None, :, None])
            window_coords_r[:, 1, :, :, :, :] = window_coords[:, 1, :, :, :, :] * torch.cos(
                sampling_angle[:, 0, :, None, :, None]) + window_coords[:, 0, :, :, :, :] * torch.sin(
                sampling_angle[:, 0, :, None, :, None])

            # system transformation: window center -> image center

            coords = window_reference + window_coords_r + sampling_offsets[:, :, :, None, :, None]

        # final offset
        sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(num_predict_total, self.attn_ws * window_num_h,
                                                                 self.attn_ws * window_num_w, 2)

        qkv = self.qkv(shortcut.permute(0, 2, 3, 1).reshape(b, -1, self.dim)).permute(0, 2, 1).reshape(b, -1, h,
                                                                                                       w).reshape(b, 3,
                                                                                                                  self.num_heads,
                                                                                                                  self.out_dim // self.num_heads,
                                                                                                                  h,
                                                                                                                  w).transpose(
            1, 0).reshape(3 * b * self.num_heads, self.out_dim // self.num_heads, h, w)
        # if self.shift_size > 0:
        qkv = torch.nn.functional.pad(qkv, (padding_left, padding_right, padding_top, padding_down)).reshape(3,
                                                                                                             b * self.num_heads,
                                                                                                             self.out_dim // self.num_heads,
                                                                                                             h + padding_td,
                                                                                                             w + padding_lr)
        # else:
        #     qkv = qkv.reshape(3, b*self.num_heads, self.dim // self.num_heads, h, w)
        q, k, v = qkv[0], qkv[1], qkv[2]  # b*self.num_heads, self.out_dim // self.num_heads, H，W

        k_selected = F.grid_sample(
            k.reshape(num_predict_total, self.out_dim // self.num_heads // self.num_deform, h + padding_td,
                      w + padding_lr),
            grid=sample_coords, padding_mode='zeros', align_corners=True
        ).reshape(b * self.num_heads, self.out_dim // self.num_heads, h + padding_td, w + padding_lr)
        v_selected = F.grid_sample(
            v.reshape(num_predict_total, self.out_dim // self.num_heads // self.num_deform, h + padding_td,
                      w + padding_lr),
            grid=sample_coords, padding_mode='zeros', align_corners=True
        ).reshape(b * self.num_heads, self.out_dim // self.num_heads, h + padding_td, w + padding_lr)

        q = q.reshape(b, self.num_heads, self.out_dim // self.num_heads, window_num_h, self.ws, window_num_w,
                      self.ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b * window_num_h * window_num_w, self.num_heads,
                                                                    self.ws * self.ws, self.out_dim // self.num_heads)
        k = k_selected.reshape(b, self.num_heads, self.out_dim // self.num_heads, window_num_h, self.attn_ws,
                               window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(
            b * window_num_h * window_num_w, self.num_heads, self.attn_ws * self.attn_ws,
            self.out_dim // self.num_heads)
        v = v_selected.reshape(b, self.num_heads, self.out_dim // self.num_heads, window_num_h, self.attn_ws,
                               window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(
            b * window_num_h * window_num_w, self.num_heads, self.attn_ws * self.attn_ws,
            self.out_dim // self.num_heads)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        dots = calc_rel_pos_spatial(dots, q, (self.ws, self.ws), (self.attn_ws, self.attn_ws), self.rel_pos_h,
                                    self.rel_pos_w)

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.attn_ws * self.attn_ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        out = attn @ v

        out = rearrange(out, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads, b=b,
                        hh=window_num_h, ww=window_num_w, ws1=self.ws, ws2=self.ws)
        # if self.shift_size > 0:
        # out = torch.masked_select(out, self.select_mask).reshape(b, -1, h, w)
        out = out[:, :, padding_top:h + padding_top, padding_left:w + padding_left]

        out = out.permute(0, 2, 3, 1).reshape(B, H * W, -1)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def _clip_grad(self, grad_norm):
        # print('clip grads of the model for selection')
        nn.utils.clip_grad_norm_(self.sampling_offsets.parameters(), grad_norm)
        nn.utils.clip_grad_norm_(self.sampling_scales.parameters(), grad_norm)

    def _reset_parameters(self):
        if self.learnable:
            nn.init.constant_(self.sampling_offsets[-1].weight, 0.)
            nn.init.constant_(self.sampling_offsets[-1].bias, 0.)
            nn.init.constant_(self.sampling_scales[-1].weight, 0.)
            nn.init.constant_(self.sampling_scales[-1].bias, 0.)

    def flops(self, ):
        N = self.ws * self.ws
        M = self.attn_ws * self.attn_ws
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * M
        #  x = (attn @ v)
        flops += self.num_heads * N * M * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        h, w = self.img_size[0] + self.shift_size + self.padding_bottom, self.img_size[
            1] + self.shift_size + self.padding_right
        flops *= (h / self.ws * w / self.ws)

        # for sampling
        flops_sampling = 0
        if self.learnable:
            # pooling
            flops_sampling += h * w * self.dim
            # regressing the shift and scale
            flops_sampling += 2 * (h / self.ws + w / self.ws) * self.num_heads * 2 * self.dim
            # calculating the coords
            flops_sampling += h / self.ws * self.attn_ws * w / self.ws * self.attn_ws * 2
        # grid sampling attended features
        flops_sampling += h / self.ws * self.attn_ws * w / self.ws * self.attn_ws * self.dim

        flops += flops_sampling

        return flops


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, window=False, restart_regression=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if not window:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        else:
            self.attn = RotatedVariedSizeWindowAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
                restart_regression=restart_regression)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, H, W):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class LoRALayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features=in_features, out_features=out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class SVDLinear(nn.Linear, LoRALayer):
    # SVD-based adaptation implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_features))
            )
            self.lora_E = nn.Parameter(
                self.weight.new_zeros(r, 1)
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features, r))
            )
            self.ranknum = nn.Parameter(
                self.weight.new_zeros(1), requires_grad=False
            )
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha if self.lora_alpha > 0 else float(self.r)
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.ranknum.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear
            # and E (singular values) for zero
            nn.init.zeros_(self.lora_E)
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)

    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(
                    self.lora_B @ (self.lora_A * self.lora_E)
                ) * self.scaling / (self.ranknum + 1e-5)
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += T(
                    self.lora_B @ (self.lora_A * self.lora_E)
                ) * self.scaling / (self.ranknum + 1e-5)
            self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (
                                  self.lora_dropout(x) @ (self.lora_A * self.lora_E).T @ self.lora_B.T
                          ) * self.scaling / (self.ranknum + 1e-5)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

class RankAllocator(object):
    """
    The RankAllocator for AdaLoRA Model that will be called every training step.
    Paper: https://openreview.net/pdf?id=lq62uWRJjiY

    Args:
        model: the model that we apply AdaLoRA to.
        lora_r (`int`): The initial rank for each incremental matrix.
        target_rank (`int`): The target average rank of incremental matrix.
        init_warmup (`int`): The steps of initial fine-tuning warmup.
        final_warmup (`int`): The step of final fine-tuning.
        mask_interval (`int`): The time internval between two budget allocations.
        beta1 (`float`): The hyperparameter of EMA for sensitivity smoothing.
        beta2 (`float`): The hyperparameter of EMA for undertainty quantification.
        total_step (`int`): The total training steps, correctly configured before training.
        target_total_rank (`Optinal[int]`): The speficified final total rank.
        tb_writter (`SummaryWriter`): Tensorboard SummaryWriter.
        tb_writter_loginterval (`int`): The logging interval of SummaryWriter.
    """

    def __init__(
            self, model,
            lora_r: int,
            target_rank: int,
            init_warmup: int,
            final_warmup: int,
            mask_interval: int,
            beta1: float,
            beta2: float,
            total_step: Optional[int] = None,
            target_total_rank: Optional[int] = None,
            tb_writter=None,
            tb_writter_loginterval: int = 500,
    ):
        self.ave_target_rank = target_rank
        self.target_rank = target_total_rank
        self.lora_init_rank = lora_r
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}
        self.rank_pattern = {}
        self.get_lora_param_name()

        self.tb_writter = tb_writter
        self.log_interval = tb_writter_loginterval

        assert (self.beta1 < 1 and self.beta1 > 0)
        assert (self.beta2 < 1 and self.beta2 > 0)

    def set_total_step(self, total_step: int):
        # Set total step number
        self.total_step = total_step
        assert self.total_step > self.initial_warmup + self.final_warmup

    def get_rank_pattern(self):
        # Return rank pattern
        return self.rank_pattern

    def get_lora_param_name(self):
        # Prepare the budget scheduler
        self.name_set = set()
        self.total_rank = 0
        self.shape_dict = {}
        for n, p in self.model.named_parameters():
            if "lora_A" in n:
                name_mat = n.replace("lora_A", "%s")
                self.name_set.add(name_mat)
                self.total_rank += p.size(0)
                self.shape_dict[n] = p.shape
            if "lora_B" in n:
                self.shape_dict[n] = p.shape
        self.name_set = list(sorted(self.name_set))
        if self.target_rank is None:
            self.target_rank = self.ave_target_rank * len(self.name_set)

    def schedule_threshold(self, step: int):
        # Global budget schedule
        mask_ind = False
        target_rank = self.target_rank
        initial_warmup = self.initial_warmup
        final_warmup = self.final_warmup
        total_step = self.total_step
        self.global_step = step
        if step <= initial_warmup:
            # Initial warmup
            curr_rank = self.total_rank
            mask_ind = False
        elif step > total_step - final_warmup:
            # Final fine-tuning
            curr_rank = self.target_rank
            # Fix the rank pattern by
            # always masking the same unimportant singluar values
            mask_ind = True
        else:
            # Budget decreasing
            mul_coeff = 1 - (step - initial_warmup) / (total_step - final_warmup - initial_warmup)
            curr_rank = target_rank + (self.total_rank - target_rank) * (mul_coeff ** 3)
            curr_rank = int(curr_rank)
            mask_ind = True if step % self.mask_interval == 0 else False
        return curr_rank, mask_ind

    def update_ipt(self, model):
        for n, p in model.named_parameters():
            if "lora_" in n:
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p)
                    self.exp_avg_unc[n] = torch.zeros_like(p)
                with torch.no_grad():
                    # Calculate sensitivity
                    self.ipt[n] = (p * p.grad).abs().detach()
                    # Update sensitivity
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                          (1 - self.beta1) * self.ipt[n]
                    # Update uncertainty
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                          (1 - self.beta2) * (self.ipt[n] - self.exp_avg_ipt[n]).abs()

    def calculate_score(self, n, p=None, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = p.abs().detach().clone()
        else:
            raise ValueError("Unexcptected Metric: %s" % metric)
        return ipt_score

    def _combine_ipt(self, ipt_E, ipt_AB):
        ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
        sum_ipt = ipt_E.view(-1) + ipt_AB.view(-1)
        return sum_ipt

    def mask_to_target_rank(self, model, curr_rank):
        is_dict = {}
        combine_dict = {}
        singular_dict = {}
        # Calculate the importance score for each sub matrix
        for n, p in model.named_parameters():
            if "lora_A" in n:
                rdim, hdim_a = p.shape
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=1, keepdim=True)
                name_mat = n.replace("lora_A", "%s")
                if name_mat not in combine_dict:
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_B" in n:
                hdim_b, rdim = p.shape
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score, dim=0, keepdim=False).view(-1, 1)
                name_mat = n.replace("lora_B", "%s")
                if name_mat not in combine_dict:
                    combine_dict[name_mat] = [comb_ipt]
                else:
                    combine_dict[name_mat].append(comb_ipt)
            if "lora_E" in n:
                ipt_score = self.calculate_score(n, p=p, metric="ipt")
                name_mat = n.replace("lora_E", "%s")
                singular_dict[name_mat] = ipt_score

        # Combine the importance scores
        all_is = []
        for name_mat in combine_dict:
            ipt_E = singular_dict[name_mat]
            ipt_AB = torch.cat(combine_dict[name_mat], dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            name_E = name_mat % "lora_E"
            is_dict[name_E] = sum_ipt.view(-1, 1)
            all_is.append(sum_ipt.view(-1))

        # Calculate the masking threshold
        mask_threshold = torch.kthvalue(torch.cat(all_is), (self.total_rank - curr_rank))[0].item()

        # Mask out unimportant singular values
        with torch.no_grad():
            curr_sum_rank = 0
            sum_param = 0
            for n, p in model.named_parameters():
                if "lora_E" in n:
                    p.data.masked_fill_(is_dict[n] <= mask_threshold, 0.0)
                    ranknum = (is_dict[n] > mask_threshold).sum().item()

                    if self.tb_writter is not None and self.global_step % self.log_interval == 0:
                        self.tb_writter.add_scalar("Ranknum/%s" % (n,), ranknum, self.global_step)
                        self.rank_pattern[n] = ranknum
                        curr_sum_rank += ranknum
                        sum_param += ranknum * self.shape_dict[n.replace("lora_E", "lora_A")][1]
                        sum_param += ranknum * self.shape_dict[n.replace("lora_E", "lora_B")][0]

            if self.tb_writter is not None and self.global_step % self.log_interval == 0:
                self.tb_writter.add_scalar("Budget/total_rank", curr_sum_rank, self.global_step)
                self.tb_writter.add_scalar("Budget/mask_threshold", mask_threshold, self.global_step)
                self.tb_writter.add_scalar("Budget/sum_param", sum_param, self.global_step)

        return mask_threshold

    def update_and_mask(self, model, global_step):
        if global_step < self.total_step - self.final_warmup:
            # Update importance scores element-wise
            self.update_ipt(model)
            # do not update ipt during final fine-tuning
        # Budget schedule
        curr_rank, mask_ind = self.schedule_threshold(global_step)
        if mask_ind:
            # Mask to target budget
            mask_threshold = self.mask_to_target_rank(model, curr_rank)
        else:
            mask_threshold = None
        self._maybe_tb_writter_log(model)
        return curr_rank, mask_threshold

    def _maybe_tb_writter_log(self, model):
        if self.tb_writter is not None and self.global_step % self.log_interval == 0:
            with torch.no_grad():
                regu_loss = []
                for n, p in model.named_parameters():
                    if "lora_A" in n or "lora_B" in n:
                        mat = p.data.detach().clone()
                        mat_cov = mat @ mat.T if "lora_A" in n else mat.T @ mat
                        I = torch.eye(*mat_cov.size(), out=torch.empty_like(mat_cov))
                        I.requires_grad = False
                        orth_regu = torch.norm(mat_cov - I, p="fro")
                        regu_loss.append(orth_regu.item())
                        self.tb_writter.add_scalar(
                            "Orth_regu_loss/%s" % n, orth_regu.item(), self.global_step
                        )
                self.tb_writter.add_scalar(
                    "train/orth_regu_loss", sum(regu_loss) / len(regu_loss), self.global_step
                )

def compute_orth_regu(model, regu_weight=0.1):
    # The function to compute orthongonal regularization for SVDLinear in `model`.
    regu_loss, num_param = 0., 0
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            para_cov = p @ p.T if "lora_A" in n else p.T @ p
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))
            I.requires_grad = False
            regu_loss += torch.norm(para_cov - I, p="fro")
            num_param += 1
    return regu_weight * regu_loss / num_param

class RVSA_MTP(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, init_values=None, use_checkpoint=False,
                 use_abs_pos_emb=False, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 out_indices=[11], interval=3, pretrained=None, restart_regression=True,
                 D2LoRA_r = 0, D2LoRA_alpha=0):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.in_chans = in_chans

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.out_indices = out_indices

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_checkpoint = use_checkpoint

        # MHSA after interval layers
        # WMHSA in other layers

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values,
                window_size=(7, 7) if ((i + 1) % interval != 0) else self.patch_embed.patch_shape,
                window=((i + 1) % interval != 0),
                restart_regression=restart_regression,
                 )
            for i in range(depth)])

        if D2LoRA_r > 0:
            # self.blocks.requires_grad=False
            print('use D2LoRA', D2LoRA_r)
            for i in range(depth):
                self.blocks[i].attn.qkv = D2LoRALinear(in_features=embed_dim, out_features=int(embed_dim*3),
                                                        r=D2LoRA_r, alpha=D2LoRA_alpha, dropout=0)
                self.blocks[i].attn.proj = D2LoRALinear(in_features=embed_dim, out_features=embed_dim,
                                                         r=D2LoRA_r, alpha=D2LoRA_alpha, dropout=0)
                self.blocks[i].mlp.fc1 = D2LoRALinear(in_features=embed_dim, out_features=int(embed_dim*mlp_ratio),
                                                       r=D2LoRA_r, alpha=D2LoRA_alpha, dropout=0)
                self.blocks[i].mlp.fc2 = D2LoRALinear(in_features=int(embed_dim * mlp_ratio),out_features=embed_dim,
                                                       r=D2LoRA_r, alpha=D2LoRA_alpha, dropout=0)


        self.interval = interval

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        # self.norm = norm_layer(embed_dim)

        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Identity()

            self.fpn3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
            )

        self.apply(self._init_weights)
        self.fix_init_weight()
        self.pretrained = pretrained

        self.out_channels = [embed_dim, embed_dim, embed_dim, embed_dim]

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        # print('MTP have initialed')
        pretrained = self.pretrained

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)

            checkpoint = torch.load(pretrained, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']########!!!!!!!!!!!!!!!!!!!!!实际分支
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint


            # print('$$$$$$$$$$$$$$$$$')
            # print(state_dict.keys())

            # print('#################')
            # print(self.state_dict().keys())

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # for MoBY, load model of online branch
            if sorted(list(state_dict.keys()))[0].startswith('encoder'):
                state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

            # # remove patch embed when inchan != 3

            # if self.in_chans != 3:
            #     for k in list(state_dict.keys()):
            #         if 'patch_embed.proj' in k:
            #             del state_dict[k]

            # rel pos of fill attn

            full_attn_rel_pos_h = None
            for name, param in self.named_parameters():
                if 'attn.full_attn_rel_pos_h' in name:
                    new_rel_sp_dim = param.shape[0]
                    new_head_dim = param.shape[1]
                    full_attn_rel_pos_h = True
                    break

            if full_attn_rel_pos_h:

                for k in list(state_dict.keys()):

                    if 'full_attn_rel_pos_h' in k or 'full_attn_rel_pos_w' in k:
                        old_rel_sp_dim = state_dict[k].shape[0]

                        old_head_dim = state_dict[k].shape[1]

                        old_rel_pos = state_dict[k]

                        old_rel_pos = old_rel_pos.reshape(1, 1, old_rel_sp_dim, old_head_dim)

                        new_rel_pos = torch.nn.functional.interpolate(
                            old_rel_pos, size=(new_rel_sp_dim, new_head_dim),
                            mode='bicubic', align_corners=False)

                        new_rel_pos = new_rel_pos.squeeze()

                        state_dict[k] = new_rel_pos

            rank, _ = get_dist_info()
            if 'pos_embed' in state_dict:
                pos_embed_checkpoint = state_dict['pos_embed']

                embedding_size = pos_embed_checkpoint.shape[-1]
                H, W = self.patch_embed.patch_shape
                num_patches = self.patch_embed.num_patches

                if 'cls_token' in state_dict.keys():
                    num_extra_tokens = 1
                else:
                    num_extra_tokens = 0
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    if rank == 0:
                        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, H, W))
                    # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(H, W), mode='bicubic', align_corners=False)
                    new_pos_embed = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    state_dict['pos_embed'] = new_pos_embed
                else:
                    state_dict['pos_embed'] = pos_embed_checkpoint[:, num_extra_tokens:]

            msg = self.load_state_dict(state_dict, False)
            if rank == 0:
                print(msg)

        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):

        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, Hp, Wp)
            else:
                x = blk(x, Hp, Wp)    ###[b,1024,768]

            if i in self.out_indices:
                features.append(x)

        features = list(map(lambda x: x.permute(0, 2, 1).reshape(B, -1, Hp, Wp), features))

        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(ops)):
            features[i] = ops[i](features[i])

        return tuple(features)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class Seg_RVSA_MTP(nn.Module):
    def __init__(self):
        super().__init__()
        norm_cfg = dict(type='BN', requires_grad=True)
        self.backbone = RVSA_MTP(
             img_size=512,
             patch_size=16,
             drop_path_rate=0.3,
             out_indices=[3, 5, 7, 11],
             embed_dim=768,
             depth=12,
             num_heads=12,
             mlp_ratio=4,
             qkv_bias=True,
             qk_scale=None,
             drop_rate=0.,
             attn_drop_rate=0.,
             use_checkpoint=False,
             use_abs_pos_emb=True,
             pretrained='/home/wuyi123/.cache/torch/hub/checkpoints/last_vit_b_rvsa_ss_is_rd_pretrn_model_encoder.pth',
             LoRA_r = 8,
             LoRA_alpha=16
                        )
        self.backbone.init_weights()
        self.decoderhead = UPerHead(
            in_channels=[768, 768, 768, 768],
            num_classes=6,
            ignore_index=255,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            align_corners=False)

        # l = 0
        # for i in range(12):
        #     if self.backbone.blocks[i].attn.qkv.weight.requires_grad ==False:l+=1
        #     if self.backbone.blocks[i].attn.proj.weight.requires_grad ==False:l+=1
        #     if self.backbone.blocks[i].mlp.fc1.weight.requires_grad ==False:l+=1
        #     if self.backbone.blocks[i].mlp.fc2.weight.requires_grad == False: l += 1
        # print(l)


    def forward(self, x):
        x=self.decoderhead(self.backbone(x))
        return F.interpolate(x,size=512,mode='bilinear')

if __name__ == "__main__":
    norm_cfg = dict(type='BN', requires_grad=True)
    model=Seg_RVSA_MTP()
    x = torch.randn(4,3,512,512)
    print(model(x).shape)