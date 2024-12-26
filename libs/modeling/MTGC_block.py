import torch
from torch import nn
from .models import register_MTGC_block
from .blocks import (TransformerBlock, MaskedConv1D, LayerNorm, get_channel_mask)

@register_MTGC_block("MTGC_block")
class MTGC_Block(nn.Module):
    def __init__(
            self,
            in_channel,
            n_embd,
            n_embd_ks,
            max_seq_len,
            num_classes,
            path_pdrop,
            n_head=1,
            scale_factor=2,
            sum_gra=6,
            use_channel_att=True
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.num_classes = num_classes
        self.relu = nn.ReLU(inplace=True)
        self.use_channel_att = use_channel_att
        self.diff_seq_len = []

        for idx in range(sum_gra):
            self.diff_seq_len.append(int(max_seq_len / (2 ** idx)))

        self.feature_expand = MaskedConv1D(in_channel, n_embd * self.num_classes,
                                           n_embd_ks, stride=1, padding=n_embd_ks // 2, bias=False)

        self.cooccur_branch = TransformerBlock(n_embd, n_head, n_hidden=n_embd, path_pdrop=path_pdrop)
        self.temporal_branch = TransformerBlock(n_embd, n_head, n_hidden=n_embd, path_pdrop=path_pdrop)

        self.feature_squeeze = MaskedConv1D(n_embd * self.num_classes, in_channel, n_embd_ks,
                                            stride=1, padding=n_embd_ks // 2, bias=False)
        self.C2F_attention = TransformerBlock(in_channel, n_head=4, path_pdrop=path_pdrop)

        self.F2C_maxpooling = nn.ModuleList()
        self.F2C_attention_block = nn.ModuleList()
        self.C2F_linear = nn.ModuleList()

        for idx in range(sum_gra - 1):
            self.F2C_maxpooling.append(nn.MaxPool1d(
                n_embd_ks, stride=2, padding=n_embd_ks // 2)
            )
            self.F2C_attention_block.append(TransformerBlock(
                in_channel, n_head,
                n_ds_strides=(1, 1),
                path_pdrop=path_pdrop,
                )
            )
            self.C2F_linear.append(nn.Linear(self.diff_seq_len[idx + 1], self.diff_seq_len[idx]))

        if self.use_channel_att:
            self.feature_tem_press = MaskedConv1D(in_channel, n_embd,
                                                  n_embd_ks, stride=1, padding=n_embd_ks // 2, bias=False)
            self.feature_tem_block = nn.ModuleList()
            self.channel_att = nn.ModuleList()
            self.channel_mask = get_channel_mask(n_embd)

            for idx in range(sum_gra):
                self.feature_tem_block.append(
                    nn.Linear(self.diff_seq_len[idx], self.diff_seq_len[idx] * self.num_classes))
                self.channel_att.append(
                    TransformerBlock(self.diff_seq_len[idx], n_head, n_hidden=n_embd, path_pdrop=path_pdrop))

        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, gra_feature, fpn_masks):
        out_features_list = tuple()
#------------------------------------------C2F process-------------------------------------------------
        for idx, (features, mask) in enumerate(zip(gra_feature, fpn_masks)):
            if idx != (len(fpn_masks) - 1):
                B, D, T = features.shape
                linear_trans_features = self.C2F_linear[idx](gra_feature[idx+1].view(B * D, -1)) #Zk+1 shape:[16*1024, 256/(2 ** (k+1))]->[16*1024, 256/(2 ** k)]
                linear_trans_features = self.relu(linear_trans_features)
                linear_trans_features = linear_trans_features.reshape(B, D, T)                 #shape: [16*1024, 256/(2 ** k)] -> [16, 1024, 256/(2 ** k)]
                features, _ = self.C2F_attention(features, linear_trans_features, mask)        #shape: [16, 1024, 256/(2 ** k)]
# ------------------------------------------C2F process-------------------------------------------------

            if self.use_channel_att:
                in_features, _ = self.feature_tem_press(features, mask)
                in_features = self.relu(in_features)
                in_features = in_features.transpose(1, 2)
                in_features, _ = self.channel_att[idx](in_features, in_features, self.channel_mask)
                in_features = in_features.transpose(1, 2)
                in_features = in_features.view(-1, in_features.shape[-1])

                features_exp_tem = self.feature_tem_block[idx](in_features)
                features_exp_tem = self.relu(features_exp_tem).view(features.shape[0], self.num_classes, -1,
                                                                    features.shape[-1]).contiguous()

                features_exp_dim, mask = self.feature_expand(features, mask)
                features_exp_dim = self.relu(features_exp_dim).view(features.shape[0], self.num_classes, -1,
                                                                    features.shape[-1]).contiguous()
                features_exp = features_exp_tem + features_exp_dim
            else:
                features_exp, mask = self.feature_expand(features, mask)
                features_exp = self.relu(features_exp).view(features.shape[0], self.num_classes, -1, features.shape[-1]).contiguous()

            B, C, H, T = features_exp.size()

            temp_feat = features_exp.view(-1, H, T)
            temp_mask = mask.repeat(C, 1, 1)
            temp_output, _ = self.temporal_branch(temp_feat, temp_feat, temp_mask)
            temp_output = temp_output.view(B, C, H, T).contiguous()

            coo_feat = features_exp.transpose(1, 3).contiguous().view(-1, H, C)
            coo_mask = mask.flatten()
            coo_output, _ = self.cooccur_branch(coo_feat, coo_feat, coo_mask)
            coo_output = coo_output.view(B, T, H, C).contiguous()

            output = temp_output + coo_output.transpose(1, 3).contiguous()

            out_features = output.view(output.shape[0], -1, output.shape[-1])
            out_features, mask = self.feature_squeeze(out_features, mask)

            out_features_list += (out_features,)

# ------------------------------------------F2C process-------------------------------------------------
        out_list = tuple()
        up_feature = out_features_list[0]  #shape: [16, 1024, 256]
        out_list += (up_feature,)

        for idx in range(len(self.F2C_maxpooling)):
            up_feature = self.F2C_maxpooling[idx](up_feature)
            up_feature, _ = self.F2C_attention_block[idx](up_feature, out_features_list[idx + 1], fpn_masks[idx + 1])
            out_list += (out_features_list[idx + 1] + up_feature,)
            up_feature = out_features_list[idx + 1] + up_feature
# ------------------------------------------F2C process-------------------------------------------------
        return out_list, fpn_masks

        
