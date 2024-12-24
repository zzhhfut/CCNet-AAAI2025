import os
import json
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from typing import List
from typing import Tuple
from typing import Dict


def remove_duplicate_annotations(ants, tol=1e-3):
    valid_events = []
    for event in ants:
        s, e, l = event['segment'][0], event['segment'][1], event['label_id']
        valid = True
        for p_event in valid_events:
            if ((abs(s-p_event['segment'][0]) <= tol)
                and (abs(e-p_event['segment'][1]) <= tol)
                and (l == p_event['label_id'])
            ):
                valid = False
                break
        if valid:
            valid_events.append(event)
    return valid_events


def load_gt_seg_from_json(json_file, split=None, label='label_id', label_offset=0):
    with open(json_file, "r", encoding="utf8") as f:
        json_db = json.load(f)
    json_db = json_db['database']

    vids, starts, stops, labels = [], [], [], []
    for k, v in json_db.items():

        if (split is not None) and v['subset'].lower() != split:
            continue

        ants = remove_duplicate_annotations(v['annotations'])
        vids += [k] * len(ants)

        for event in ants:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):

                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset**i + int(x)
            else:

                label_id = int(event[label])
            labels += [label_id]

    gt_base = pd.DataFrame({
        'video-id' : vids,
        't-start' : starts,
        't-end': stops,
        'label': labels
    })

    return gt_base


def load_pred_seg_from_json(json_file, label='label_id', label_offset=0):
    with open(json_file, "r", encoding="utf8") as f:
        json_db = json.load(f)
    json_db = json_db['database']

    vids, starts, stops, labels, scores = [], [], [], [], []
    for k, v, in json_db.items():
        vids += [k] * len(v)
        for event in v:
            starts += [float(event['segment'][0])]
            stops += [float(event['segment'][1])]
            if isinstance(event[label], (Tuple, List)):
                label_id = 0
                for i, x in enumerate(event[label][::-1]):
                    label_id += label_offset**i + int(x)
            else:
                label_id = int(event[label])
            labels += [label_id]
            scores += [float(event['scores'])]

    pred_base = pd.DataFrame({
        'video-id' : vids,
        't-start' : starts,
        't-end': stops,
        'label': labels,
        'score': scores
    })

    return pred_base


class ANETdetection(object):
    def __init__(
        self,
        ant_file,
        split=None,
        tiou_thresholds=np.linspace(0.1, 0.5, 5),
        label='label_id',
        label_offset=0,
        num_workers=8,
        dataset_name=None,
    ):

        self.tiou_thresholds = tiou_thresholds
        self.ap = None
        self.num_workers = num_workers
        if dataset_name is not None:
            self.dataset_name = dataset_name
        else:
            self.dataset_name = os.path.basename(ant_file).replace('.json', '')

        self.split = split
        self.ground_truth = load_gt_seg_from_json(
            ant_file, split=self.split, label=label, label_offset=label_offset)

        self.activity_index = {j: i for i, j in enumerate(sorted(self.ground_truth['label'].unique()))}
        self.ground_truth['label']=self.ground_truth['label'].replace(self.activity_index)

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        try:
            res = prediction_by_label.get_group(cidx).reset_index(drop=True)
            return res
        except:
            print('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self, preds):
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = preds.groupby('label')

        results = Parallel(n_jobs=self.num_workers)(
            delayed(compute_average_precision_detection)(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                tiou_thresholds=self.tiou_thresholds,
            ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:,cidx] = results[i]

        return ap

    def evaluate(self, preds, verbose=True):
        if isinstance(preds, pd.DataFrame):
            assert 'label' in preds
        elif isinstance(preds, str) and os.path.isfile(preds):
            preds = load_pred_seg_from_json(preds)
        elif isinstance(preds, Dict):
            preds = pd.DataFrame({
                'video-id' : preds['video-id'],
                't-start' : preds['t-start'].tolist(),
                't-end': preds['t-end'].tolist(),
                'label': preds['label'].tolist(),
                'score': preds['score'].tolist()
            })
        self.ap = None

        preds['label'] = preds['label'].replace(self.activity_index)

        self.ap = self.wrapper_compute_average_precision(preds)
        mAP = self.ap.mean(axis=1)
        average_mAP = mAP.mean()

        if verbose:

            print('[RESULTS] Action detection results on {:s}.'.format(
                self.dataset_name)
            )
            block = ''
            for tiou, tiou_mAP in zip(self.tiou_thresholds, mAP):
                block += '\n|tIoU = {:.2f}: mAP = {:.2f} (%)'.format(tiou, tiou_mAP*100)
            print(block)
            print('Avearge mAP: {:.2f} (%)'.format(average_mAP*100))

        return mAP, average_mAP


def compute_average_precision_detection(
    ground_truth,
    prediction,
    tiou_thresholds=np.linspace(0.1, 0.5, 5)
):
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    ground_truth_gbvn = ground_truth.groupby('video-id')

    for idx, this_pred in prediction.iterrows():

        try:
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap


def segment_iou(target_segment, candidate_segments):
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    segments_intersection = (tt2 - tt1).clip(0)
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
                     + (target_segment[1] - target_segment[0]) - segments_intersection

    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


def interpolated_prec_rec(prec, rec):
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap
