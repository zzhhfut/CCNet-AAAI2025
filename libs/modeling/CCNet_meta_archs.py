import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import (register_CCNet_meta_arch, make_CCNet_backbone,
                    make_MTGC_block)
from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss

from ..utils import batched_nms


class CCNet_ClsHead(nn.Module):
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = [],
    ):
        super().__init__()
        self.act = act_layer()

        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()

        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.cls_head = MaskedConv1D(
                feat_dim, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )

        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits, )

        return out_logits


class CCNet_RegHead(nn.Module):
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        class_aware=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()
        self.num_classes = num_classes

        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim

            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        if class_aware:
            self.offset_head = MaskedConv1D(
                    feat_dim, 2*num_classes, kernel_size,
                    stride=1, padding=kernel_size//2
                )
        else:
            self.offset_head = MaskedConv1D( 
                feat_dim, 2, kernel_size,
                stride=1, padding=kernel_size//2
            )


    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)), ) 

        return out_offsets


@register_CCNet_meta_arch("CCNet_base_TCG_Transformer")
class CCNet_CMCC_MTGC(nn.Module):
    def __init__(
        self,
        backbone_arch,
        scale_factor,
        input_dim_V,
        input_dim_A,
        max_seq_len,
        n_head,
        embd_kernel_size,
        embd_dim,
        embd_with_ln,
        head_dim,
        regression_range,
        head_num_layers,
        head_kernel_size,
        head_with_ln,
        num_classes,
        train_cfg,
        test_cfg,
        class_aware,
    ):
        super().__init__()
        self.fpn_strides = [scale_factor**i for i in range(backbone_arch[-1]+1)]
        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        self.num_classes = num_classes
        self.class_aware = class_aware

        self.max_seq_len = max_seq_len
        max_div_factor = 1
        for l, stride in enumerate(self.fpn_strides):
            assert max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']

        self.backbone = make_CCNet_backbone(
            'CCNet_base_Transformer',
            **{
                'n_in_V' : input_dim_V,
                'n_in_A' : input_dim_A,
                'n_embd' : embd_dim,
                'n_head': n_head,
                'n_embd_ks': embd_kernel_size,
                'max_len': max_seq_len,
                'arch' : backbone_arch,
                'scale_factor' : scale_factor,
                'with_ln' : embd_with_ln,
                'attn_pdrop' : 0.0,
                'proj_pdrop' : self.train_dropout,
                'path_pdrop' : self.train_droppath,
            }
        )

        self.MTGC_block = make_MTGC_block(
            'MTGC_block',
            **{
                'in_channel' : embd_dim*2,
                'n_embd' : 128,
                'n_embd_ks' : embd_kernel_size,
                'max_seq_len' : max_seq_len,
                'num_classes' : self.num_classes,
                'path_pdrop' : self.train_droppath,
            }
        )

        self.cls_head = CCNet_ClsHead(
            embd_dim*2,
            head_dim, self.num_classes,  
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )
        self.reg_head = CCNet_RegHead(
            embd_dim*2,
            head_dim, self.num_classes, 
            len(self.fpn_strides),
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln,
            class_aware=self.class_aware
        )

        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, video_list):
        batched_inputs_V, batched_inputs_A, batched_masks = self.preprocessing(video_list)

        feats_V, feats_A, masks = self.backbone(batched_inputs_V, batched_inputs_A, batched_masks)

        gra_feature = [torch.cat((V, A), 1) for _, (V, A) in enumerate(zip(feats_V, feats_A))]

        gra_feature,  _ = self.MTGC_block(gra_feature, masks)

        out_cls_logits = self.cls_head(gra_feature, masks)
        out_offsets = self.reg_head(gra_feature, masks)

        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]

        if self.class_aware:
            out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
            out_offsets = [x.view(x.shape[0], x.shape[1], self.num_classes, -1).contiguous() for x in out_offsets]
        else:
            out_offsets = [x.permute(0, 2, 1) for x in out_offsets]

        fpn_masks = [x.squeeze(1) for x in masks]

        if self.training:
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            gt_offsets = [x['gt_offsets'] for x in video_list]
            gt_cls_labels = [x['gt_cls_labels'] for x in video_list]

            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets
            )
            return losses

        else:
            points = video_list[0]['points']
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets
            )
            return results

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        feats_visual = [x['feats']['visual'] for x in video_list]
        feats_audio = [x['feats']['audio'] for x in video_list]
        feats_lens = torch.as_tensor([feat_visual.shape[-1] for feat_visual in feats_visual])
        max_len = feats_lens.max(0).values.item() 

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            max_len = self.max_seq_len
        else:
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride

        batch_shape_visual = [len(feats_visual), feats_visual[0].shape[0], max_len]
        batched_inputs_visual = feats_visual[0].new_full(batch_shape_visual, padding_val)
        for feat_visual, pad_feat_visual in zip(feats_visual, batched_inputs_visual):
            pad_feat_visual[..., :feat_visual.shape[-1]].copy_(feat_visual)

        batch_shape_audio = [len(feats_audio), feats_audio[0].shape[0], max_len]
        batched_inputs_audio = feats_audio[0].new_full(batch_shape_audio, padding_val)
        for feat_audio, pad_feat_audio in zip(feats_audio, batched_inputs_audio):
            pad_feat_audio[..., :feat_audio.shape[-1]].copy_(feat_audio) 

        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        batched_inputs_visual = batched_inputs_visual.to(self.device)
        batched_inputs_audio = batched_inputs_audio.to(self.device)
        
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs_visual, batched_inputs_audio, batched_masks

    def losses(
        self, fpn_masks,
        out_cls_logits, out_offsets,
        gt_cls_labels, gt_offsets
    ):
        valid_mask = torch.cat(fpn_masks, dim=1)

        gt_cls = torch.stack(gt_cls_labels)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)

        pred_offsets = torch.cat(out_offsets, dim=1)[pos_mask] 
        gt_offsets = torch.stack(gt_offsets)[pos_mask] 

        num_pos = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        gt_target = gt_cls[valid_mask]

        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='sum'
        )
        cls_loss /= self.loss_normalizer

        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum',
                class_aware=self.class_aware
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        final_loss = cls_loss + reg_loss * loss_weight
        return {'cls_loss'   : cls_loss,
                'reg_loss'   : reg_loss,
                'final_loss' : final_loss}

    @torch.no_grad()
    def inference(
        self,
        video_list,
        points, fpn_masks,
        out_cls_logits, out_offsets
    ):
        results = []

        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
            zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid
            )

            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
        self,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
    ):
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        for cls_i, offsets_i, pts_i, mask_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks
            ):
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1] 
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0] 

            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            pt_idxs =  torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor'
            ) 
            cls_idxs = torch.fmod(topk_idxs, self.num_classes) 

            if self.class_aware:
                offsets_i = offsets_i.view(-1, offsets_i.shape[-1]).contiguous() 
                offsets = offsets_i[topk_idxs] 

            else:
                offsets = offsets_i[pt_idxs] 

            pts = pts_i[pt_idxs]

            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]
            pred_segs = torch.stack((seg_left, seg_right), -1) 

            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments' : segs_all,
                   'scores'   : scores_all,
                   'labels'   : cls_idxs_all}

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        processed_results = []
        for results_per_vid in results:
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration'] 
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            if self.test_nms_method != 'none':
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms = (self.test_nms_method == 'soft'),
                    multiclass = self.test_multiclass_nms,
                    sigma = self.test_nms_sigma,
                    voting_thresh = self.test_voting_thresh
                )
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                segs[segs<=0.0] *= 0.0
                segs[segs>=vlen] = segs[segs>=vlen] * 0.0 + vlen
            processed_results.append(
                {'video_id' : vidx,
                 'segments' : segs,
                 'scores'   : scores,
                 'labels'   : labels}
            )

        return processed_results
