import torch
from torch import nn
from torch.nn import functional as F

from .models import register_CCNet_backbone
from .blocks import (get_sinusoid_encoding, TransformerBlock, 
                    MaskedConv1D, LayerNorm, TCG_block)


@register_CCNet_backbone("CCNet_base_Transformer")
class CCNet_TCG_Backbone(nn.Module):
    def __init__(
        self,
        n_in_V,
        n_in_A,
        n_embd,
        n_head,
        n_embd_ks,
        max_len,
        arch = (2, 2, 5),
        scale_factor = 2,
        with_ln = False,
        attn_pdrop = 0.0,
        proj_pdrop = 0.0,
        path_pdrop = 0.0,
    ):
        super().__init__()
        assert len(arch) == 3
        self.arch = arch
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
        self.register_buffer("pos_embd", pos_embd, persistent=False)

        self.embd_V = nn.ModuleList()
        self.embd_A = nn.ModuleList()
        self.embd_norm_V = nn.ModuleList()
        self.embd_norm_A = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels_V = n_in_V
                in_channels_A = n_in_A
            else:
                in_channels_V = n_embd
                in_channels_A = n_embd
            self.embd_V.append(MaskedConv1D(
                    in_channels_V, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            self.embd_A.append(MaskedConv1D(
                    in_channels_A, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm_V.append(
                    LayerNorm(n_embd)
                )
                self.embd_norm_A.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm_V.append(nn.Identity())
                self.embd_norm_A.append(nn.Identity())

        self.self_att_V = nn.ModuleList()
        self.self_att_A = nn.ModuleList()

        for idx in range(arch[1]):
            self.self_att_V.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
            self.self_att_A.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )

        self.ori_CMI_Visual = TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
        self.ori_CMI_Audio = TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )

        self.CMI_Visual = nn.ModuleList()
        self.CMI_Audio = nn.ModuleList()
        for idx in range(arch[2]):
            self.CMI_Visual.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )
            self.CMI_Audio.append(TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                )
            )

        self.visual_TCG = nn.ModuleList()
        self.audio_TCG = nn.ModuleList()
        for idx in range(arch[2]+1):
            if idx == 0:
                self.audio_TCG.append(TCG_block(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    )
                )
                self.visual_TCG.append(TCG_block(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    )
                )
            else:
                self.audio_TCG.append(TCG_block(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    )
                )
                self.visual_TCG.append(TCG_block(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    )
                )

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x_V, x_A, mask):
        B, C_V, T = x_V.size()
        mask_V = mask_A = mask

        for idx in range(len(self.embd_V)):
            x_V, mask_V = self.embd_V[idx](x_V, mask_V) 
            x_V = self.relu(self.embd_norm_V[idx](x_V))

            x_A, mask_A = self.embd_A[idx](x_A, mask_A)
            x_A = self.relu(self.embd_norm_A[idx](x_A))

        if self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x_V = x_V + pe[:, :, :T] * mask_V.to(x_V.dtype)
            x_A = x_A + pe[:, :, :T] * mask_A.to(x_A.dtype)
        else:
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x_V = x_V + pe[:, :, :T] * mask_V.to(x_V.dtype)
            x_A = x_A + pe[:, :, :T] * mask_A.to(x_A.dtype)

        for idx in range(len(self.self_att_V)):
            x_V, mask_V = self.self_att_V[idx](x_V, x_V, mask_V)
            x_A, mask_A = self.self_att_A[idx](x_A, x_A, mask_A)

#-----------------------first layer of CMCC------------------
        #x_V shape: [16, 512, 256], x_A shape: [16, 512, 256]
        x_Va, mask_V = self.ori_CMI_Visual(x_V, x_A, mask_V) #x_Va shape: [16, 512, 256]
        gate_a = self.audio_TCG[0](x_A, x_A, mask_V)        #gate_a shape:[16, 1, 256]
      
        x_Av, mask_A = self.ori_CMI_Audio(x_A, x_V, mask_A)  #x_Va shape: [16, 512, 256]
        gate_v = self.visual_TCG[0](x_V, x_V, mask_A)       #gate_v shape:[16, 1, 256]

        out_feats_V = tuple()
        out_feats_A = tuple()
        out_masks = tuple()

        out_feats_V += (x_Va + x_Va * gate_a, )
        out_masks += (mask_V, )
        out_feats_A += (x_Av + x_Av * gate_v, )
# ----------------------------------------------------------

        for idx in range(len(self.CMI_Visual)):
# ------------------------------------audio guide CMCC-------------------------------------
            x_V, mask_V = self.CMI_Visual[idx](out_feats_V[idx], out_feats_A[idx], mask_V)
            gate_a = self.audio_TCG[idx+1](out_feats_A[idx], out_feats_A[idx], mask_V)   #gate shape:[16, 1, 256/2**(idx+1)]
            out_feats_V += (x_V + x_V * gate_a,)                                         #output shape:[16, 512, 256/2**(idx+1)]
            out_masks += (mask_V,)
# -----------------------------------------------------------------------------------------

# -----------------------------------visual guide CMCC-------------------------------------
            x_A, mask_A = self.CMI_Audio[idx](out_feats_A[idx], out_feats_V[idx], mask_A)
            gate_v = self.visual_TCG[idx+1](out_feats_V[idx], out_feats_V[idx], mask_A)
            out_feats_A += (x_A + x_A * gate_v, )                                        #output shape:[16, 512, 256/2**(idx+1)]
# -----------------------------------------------------------------------------------------
        return out_feats_V, out_feats_A, out_masks
