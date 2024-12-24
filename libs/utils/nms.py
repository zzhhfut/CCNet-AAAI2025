import torch

import nms_1d_cpu


class NMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, segs, scores, cls_idxs,
        iou_threshold, min_score, max_num
    ):
        is_filtering_by_score = (min_score > 0)
        if is_filtering_by_score:
            valid_mask = scores > min_score
            segs, scores = segs[valid_mask], scores[valid_mask]
            cls_idxs = cls_idxs[valid_mask]
            valid_inds = torch.nonzero(
                valid_mask, as_tuple=False).squeeze(dim=1)

        inds = nms_1d_cpu.nms(
            segs.contiguous().cpu(),
            scores.contiguous().cpu(),
            iou_threshold=float(iou_threshold))
        if max_num > 0:
            inds = inds[:min(max_num, len(inds))]
        sorted_segs = segs[inds]
        sorted_scores = scores[inds]
        sorted_cls_idxs = cls_idxs[inds]
        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()


class SoftNMSop(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, segs, scores, cls_idxs,
        iou_threshold, sigma, min_score, method, max_num
    ):
        dets = segs.new_empty((segs.size(0), 3), device='cpu')
        inds = nms_1d_cpu.softnms(
            segs.cpu(),
            scores.cpu(),
            dets.cpu(),
            iou_threshold=float(iou_threshold),
            sigma=float(sigma),
            min_score=float(min_score),
            method=int(method))
        if max_num > 0:
            n_segs = min(len(inds), max_num)
        else:
            n_segs = len(inds)
        sorted_segs = dets[:n_segs, :2]
        sorted_scores = dets[:n_segs, 2]
        sorted_cls_idxs = cls_idxs[inds]
        sorted_cls_idxs = sorted_cls_idxs[:n_segs]
        return sorted_segs.clone(), sorted_scores.clone(), sorted_cls_idxs.clone()


def seg_voting(nms_segs, all_segs, all_scores, iou_threshold, score_offset=1.5):
    offset_scores = all_scores + score_offset

    num_nms_segs, num_all_segs = nms_segs.shape[0], all_segs.shape[0]
    ex_nms_segs = nms_segs[:, None].expand(num_nms_segs, num_all_segs, 2)
    ex_all_segs = all_segs[None, :].expand(num_nms_segs, num_all_segs, 2)

    left = torch.maximum(ex_nms_segs[:, :, 0], ex_all_segs[:, :, 0])
    right = torch.minimum(ex_nms_segs[:, :, 1], ex_all_segs[:, :, 1])
    inter = (right-left).clamp(min=0)

    nms_seg_lens = ex_nms_segs[:, :, 1] - ex_nms_segs[:, :, 0]
    all_seg_lens = ex_all_segs[:, :, 1] - ex_all_segs[:, :, 0]

    iou = inter / (nms_seg_lens + all_seg_lens - inter)

    seg_weights = (iou >= iou_threshold).to(all_scores.dtype) * all_scores[None, :]
    seg_weights /= torch.sum(seg_weights, dim=1, keepdim=True)
    refined_segs = seg_weights @ all_segs

    return refined_segs

def batched_nms(
    segs,
    scores,
    cls_idxs,
    iou_threshold,
    min_score,
    max_seg_num,
    use_soft_nms=True,
    multiclass=True,
    sigma=0.5,
    voting_thresh=0.75,
):
    num_segs = segs.shape[0]
    if num_segs == 0:
        return torch.zeros([0, 2]),\
               torch.zeros([0,]),\
               torch.zeros([0,], dtype=cls_idxs.dtype)

    if multiclass:
        new_segs, new_scores, new_cls_idxs = [], [], []
        for class_id in torch.unique(cls_idxs):
            curr_indices = torch.where(cls_idxs == class_id)[0]
            if use_soft_nms:
                sorted_segs, sorted_scores, sorted_cls_idxs = SoftNMSop.apply(
                    segs[curr_indices],
                    scores[curr_indices],
                    cls_idxs[curr_indices],
                    iou_threshold,
                    sigma,
                    min_score,
                    2,
                    max_seg_num
                )
            else:
                sorted_segs, sorted_scores, sorted_cls_idxs = NMSop.apply(
                    segs[curr_indices],
                    scores[curr_indices],
                    cls_idxs[curr_indices],
                    iou_threshold,
                    min_score,
                    max_seg_num
                )

            new_segs.append(sorted_segs)
            new_scores.append(sorted_scores)
            new_cls_idxs.append(sorted_cls_idxs)

        new_segs = torch.cat(new_segs)
        new_scores = torch.cat(new_scores)
        new_cls_idxs = torch.cat(new_cls_idxs)

    else:
        if use_soft_nms:
            new_segs, new_scores, new_cls_idxs = SoftNMSop.apply(
                segs, scores, cls_idxs, iou_threshold,
                sigma, min_score, 2, max_seg_num
            )
        else:
            new_segs, new_scores, new_cls_idxs = NMSop.apply(
                segs, scores, cls_idxs, iou_threshold,
                min_score, max_seg_num
            )
        if voting_thresh > 0:
            new_segs = seg_voting(
                new_segs,
                segs,
                scores,
                voting_thresh
            )

    _, idxs = new_scores.sort(descending=True)
    max_seg_num = min(max_seg_num, new_segs.shape[0])
    new_segs = new_segs[idxs[:max_seg_num]]
    new_scores = new_scores[idxs[:max_seg_num]]
    new_cls_idxs = new_cls_idxs[idxs[:max_seg_num]]
    return new_segs, new_scores, new_cls_idxs
