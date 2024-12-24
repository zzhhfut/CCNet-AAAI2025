import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

class MaskedConv1D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()

        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)

        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        B, C, T = x.size()

        assert T % self.stride == 0
        out_conv = self.conv(x)
        if self.stride > 1:

            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=T//self.stride,
                mode='nearest'
            )
        else:
            out_mask = mask.to(x.dtype)

        if out_mask.shape[-1] != out_conv.shape[-1]:
            out_mask = out_mask[:, None, None]

        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)
        if self.affine:
            out *= self.weight
            out += self.bias

        return out

def get_channel_mask(n_embd):
    channel_mask = torch.ones([1, 1, n_embd]).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    return channel_mask

def get_sinusoid_encoding(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)

class MaskedMHCA(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_qx_stride=1,
        n_kv_stride=1,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        assert (n_qx_stride == 1) or (n_qx_stride % 2 == 0)
        assert (n_kv_stride == 1) or (n_kv_stride % 2 == 0)
        self.n_qx_stride = n_qx_stride
        self.n_kv_stride = n_kv_stride

        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2

        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )

        self.query_norm = LayerNorm(self.n_embd)

        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2
        # 1d depthwise conv
        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )

        self.value_norm = LayerNorm(self.n_embd)

        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)

    def forward(self, x1, x2, mask):
        # k, v: x1
        # q: x2
        B, C, T = x1.size()

        q, qx_mask = self.query_conv(x2, mask)
        q = self.query_norm(q)
        k, kv_mask = self.key_conv(x1, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x1, mask)
        v = self.value_norm(v)

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        att = (q * self.scale) @ k.transpose(-2, -1)

        if T == mask.shape[-1]:
            att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
            att = F.softmax(att, dim=-1) 
        else:
            att = F.softmax(att, dim=-1)  
            att = att * kv_mask[:, :, :, None].to(att.dtype) 

        att = self.attn_drop(att)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)

        return out, qx_mask

class TCG_block(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_ds_strides=(1, 1),
        attn_pdrop=0.0,
        proj_pdrop=0.0,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        assert len(n_ds_strides) == 2

        self.ln11 = LayerNorm(n_embd)
        self.ln12 = LayerNorm(n_embd)

        self.n_embd = n_embd
        self.n_head = n_head
        self.n_channels = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.n_channels)

        self.n_qx_stride = n_ds_strides[0]
        self.n_kv_stride = n_ds_strides[1]
        assert (self.n_qx_stride == 1) or (self.n_qx_stride % 2 == 0)
        assert (self.n_kv_stride == 1) or (self.n_kv_stride % 2 == 0)

        kernel_size = self.n_qx_stride + 1 if self.n_qx_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2

        self.query_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )

        self.query_norm = LayerNorm(self.n_embd)

        kernel_size = self.n_kv_stride + 1 if self.n_kv_stride > 1 else 3
        stride, padding = self.n_kv_stride, kernel_size // 2

        self.key_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )
        self.key_norm = LayerNorm(self.n_embd)
        self.value_conv = MaskedConv1D(
            self.n_embd, self.n_embd, kernel_size,
            stride=stride, padding=padding, groups=self.n_embd, bias=False
        )

        self.value_norm = LayerNorm(self.n_embd)

        self.key = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.query = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.value = nn.Conv1d(self.n_embd, self.n_embd, 1)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        self.proj = nn.Conv1d(self.n_embd, self.n_embd, 1)
        self.gate_linear = MaskedConv1D(
            self.n_embd, 1, kernel_size=1, bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, mask):
        # k, v: x1
        # q: x2
        B, C, T = x1.size()
        x1 = self.ln11(x1)
        x2 = self.ln12(x2)

        q, qx_mask = self.query_conv(x2, mask)
        q = self.query_norm(q)
        k, kv_mask = self.key_conv(x1, mask)
        k = self.key_norm(k)
        v, _ = self.value_conv(x1, mask)
        v = self.value_norm(v)

        q = self.query(q)
        k = self.key(k)
        v = self.value(v)

        k = k.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        q = q.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)
        v = v.view(B, self.n_head, self.n_channels, -1).transpose(2, 3)

        att = (q * self.scale) @ k.transpose(-2, -1)

        if T == mask.shape[-1]:
            att = att.masked_fill(torch.logical_not(kv_mask[:, :, None, :]), float('-inf'))
            att = F.softmax(att, dim=-1)
        else:
            att = F.softmax(att, dim=-1)
            att = att * kv_mask[:, :, :, None].to(att.dtype)

        att = self.attn_drop(att)
        out = att @ (v * kv_mask[:, :, :, None].to(v.dtype))
        out = out.transpose(2, 3).contiguous().view(B, C, -1)

        out = self.proj_drop(self.proj(out)) * qx_mask.to(out.dtype)
        out, _ = self.gate_linear(out, qx_mask)
        out = self.sigmoid(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        n_ds_strides=(1, 1),
        n_out=None,
        n_hidden=None,
        act_layer=nn.GELU,
        attn_pdrop=0.0,
        proj_pdrop=0.0,
        path_pdrop=0.0,
    ):
        super().__init__()
        assert len(n_ds_strides) == 2
        self.ln11 = LayerNorm(n_embd)
        self.ln12 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)
        self.n_embd = n_embd

        self.attn = MaskedMHCA(
            n_embd,
            n_head,
            n_qx_stride=n_ds_strides[0],
            n_kv_stride=n_ds_strides[1],
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop
            )

        if n_ds_strides[0] > 1:
            kernel_size, stride, padding = \
                n_ds_strides[0] + 1, n_ds_strides[0], (n_ds_strides[0] + 1)//2
            self.pool_skip = nn.MaxPool1d(
                kernel_size, stride=stride, padding=padding)
        else:
            self.pool_skip = nn.Identity()

        if n_hidden is None:
            n_hidden = 4 * n_embd
        if n_out is None:
            n_out = n_embd

        self.mlp = nn.Sequential(
            nn.Conv1d(n_embd, n_hidden, 1),
            act_layer(),
            nn.Dropout(proj_pdrop, inplace=True),
            nn.Conv1d(n_hidden, n_out, 1),
            nn.Dropout(proj_pdrop, inplace=True),
        )

        if path_pdrop > 0.0:
            self.drop_path_attn = AffineDropPath(n_embd, drop_prob = path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob = path_pdrop)
        else:
            self.drop_path_attn = nn.Identity()
            self.drop_path_mlp = nn.Identity()

    def forward(self, x1, x2, mask, pos_embd=None):
        out, out_mask = self.attn(self.ln11(x1), self.ln12(x2), mask)
        out_mask_float = out_mask.to(out.dtype)
        out = self.pool_skip(x1) * out_mask_float + self.drop_path_attn(out) 
        # FFN 
        out = out + self.drop_path_mlp(self.mlp(self.ln2(out)) * out_mask_float)

        if pos_embd is not None:
            out += pos_embd * out_mask_float
        
        return out, out_mask 

class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        return x * self.scale

def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):

        return drop_path(self.scale * x, self.drop_prob, self.training)
