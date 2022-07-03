from bisect import bisect
from collections import OrderedDict, defaultdict
import math
import pickle
from typing import Dict, Mapping
from ignite.metrics import Metric
import ignite.distributed as idist
from ignite.base.mixins import Serializable
import torch
import torch.distributed as dist

from ai4code.datasets import Sample


def count_inversions(a):
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions


def kendall_tau(ground_truth, predictions):
    total_inversions = 0
    total_2max = 0  # twice the maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [
            gt.index(x) for x in pred
        ]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


class KendallTauPairwise(Metric):
    """pairwise 的 metric
    对于一个 Sampe，计算所有 cell 对其他 cell 的得分，得到一个 N x N 的矩阵（N 是 cell number）
    (n, m) 表示第 m 个 cell 排在第 n 个 cell 后面的概率。
    """
    def __init__(self, val_data: Dict[str, Sample], code_dir):
        super().__init__()
        self.val_data = val_data
        self.code_dir = code_dir
        self.epoch = 0

    def reset(self):
        self._pairwise_matrix = {}

    def update(self, output):
        pairwise_matric, sample = output
        self._pairwise_matrix[sample.id] = pairwise_matric

    def compute(self):
        self.epoch += 1
        pickle.dump(self._pairwise_matrix, open(self.code_dir / f"{self.epoch}.pairwise_matrix.pkl", "wb"))
        return self.epoch


class KendallTauNaive(Metric):
    def __init__(self, val_data: Dict[str, Sample]):
        super().__init__()
        self.val_data = val_data
        self.reset()

    def reset(self):
        self._predictions = defaultdict(dict)
        self._all_predictions = []
        self._all_targets = []
        self._submission_data = {}

    def update(self, output):
        loss, scores, sample_ids, cell_ids = output
        for score, sample_id, cell_id in zip(scores, sample_ids, cell_ids):
            self._predictions[sample_id][cell_id] = score.item()

    def compute(self):
        for sample in self.val_data.values():
            all_preds = []
            for cell_key in sample.cell_keys:
                cell_type = sample.cell_types[cell_key]
                cell_rank = sample.cell_ranks_normed[cell_key]
                if cell_type == "code":
                    # keep the original cell_rank
                    item = (cell_key, cell_rank)
                else:
                    item = (cell_key, self._predictions[sample.id][cell_key])
                all_preds.append(item)
            cell_id_predicted = [
                item[0] for item in sorted(all_preds, key=lambda x: x[1])
            ]
            self._submission_data[sample.id] = cell_id_predicted
            self._all_predictions.append(cell_id_predicted)
            self._all_targets.append(sample.orders)

        score = kendall_tau(self._all_targets, self._all_predictions)
        print("Kendall Tau: ", score)
        return score


class KendallTauWithSplits(Serializable, Metric):

    def __init__(self, val_data: Dict[str, Sample], split_len):
        super().__init__()
        self.val_data = val_data
        self.split_len = split_len
        self.reset()

    def reset(self):
        self._all_predictions = []
        self._all_targets = []
        self._raw_preds = {}
        self._predictions = []

    def state_dict(self) -> Dict:
        return self._raw_preds

    def load_state_dict(self, state_dict: Mapping) -> None:
        super().load_state_dict(state_dict)
        self._raw_preds = state_dict

    def update(self, output):
        loss, in_splits, ranks, sample_ids, cell_ids, split_ids, rank_offsets = output
        for in_split, rank, sample_id, cell_id, split_id, rank_offset in zip(in_splits, ranks, sample_ids, cell_ids, split_ids, rank_offsets):
            self._predictions.append([in_split.item(), rank.item(), sample_id, cell_id, split_id.item(), rank_offset])

    def compute(self):
        if torch.distributed.is_initialized():
            all_predictions = [None for _ in range(idist.get_world_size())]
            dist.all_gather_object(all_predictions, self._predictions)
        else:
            all_predictions = [self._predictions]

        all_predictions_dict = defaultdict(lambda: defaultdict(dict))
        for in_split, rank, sample_id, cell_id, split_id, rank_offset in [x for xs in all_predictions for x in xs]:
            all_predictions_dict[sample_id][cell_id][split_id] = [in_split, rank, sample_id, split_id, rank_offset]

        if torch.distributed.is_initialized() and idist.get_local_rank() != 0:
            return

        for sample in self.val_data.values():
            all_preds = []
            for cell_key in sample.cell_keys:
                cell_type = sample.cell_types[cell_key]
                cell_rank = sample.cell_ranks[cell_key]
                if cell_type == "code":
                    # keep the original cell_rank
                    item = (cell_key, cell_rank, sample.id)
                else:
                    # markdown cell，选出 in_split 得分最高的，取 rank + split_offset
                    item = (cell_key, all_predictions_dict[sample.id][cell_key])
                    split_results = list(all_predictions_dict[sample.id][cell_key].items())
                    in_split_result = split_results[0][1]
                    for split_id, result in split_results[1:]:
                        if result[0] > in_split_result[0]:
                            in_split_result = result

                    in_split_score, rank_normed, sample_id, split_id, rank_offset = in_split_result
                    cell_rank = math.e**rank_normed + rank_offset
                    item = (cell_key, cell_rank, in_split_score, rank_normed, sample_id, split_id)

                all_preds.append(item)
            cell_id_predicted = [
                item[0] for item in sorted(all_preds, key=lambda x: x[1])
            ]
            self._raw_preds[sample.id] = all_preds
            self._all_predictions.append(cell_id_predicted)
            self._all_targets.append(sample.orders)

        score = kendall_tau(self._all_targets, self._all_predictions)
        print("Kendall Tau: ", score)
        return score
