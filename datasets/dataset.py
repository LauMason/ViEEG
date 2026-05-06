"""
datasets/dataset.py

三层 Dataset 结构：

  SubjectDataset      — 单个 subject，用于 SD（within-subject）实验
                        持有 trial_ids / session_ids，由子类实现 get_splits()

  FullDataset         — 整个数据集所有 subject 一次性加载，用于 SI（LOSO）实验
                        持有 subject_ids，get_splits() 按 subject 边界划分

  MultiDatasetWrapper — 包裹多个 FullDataset，用于跨数据集训练
                        持有 dataset_ids + subject_ids 双重标签

工厂函数：
  build_subject_dataset(cfg, subject_id)  →  SubjectDataset 子类实例
  build_full_dataset(cfg)                 →  FullDataset 子类实例
  build_multi_dataset(cfgs)              →  MultiDatasetWrapper 实例
"""

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.deap import load_deap_subject
from datasets.dreamer import load_dreamer_subject
from datasets.mahnob import load_mahnob_subject
from datasets.split import (
    kfold_by_segment,
    leave_one_trial_out,
    kfold_by_unit,
    fixed_split_by_unit,
    leave_one_subject_out,
    group_kfold_by_subject,
)


# ══════════════════════════════════════════════════════════════════════════════
# 公共基类
# ══════════════════════════════════════════════════════════════════════════════

class EEGDatasetBase(Dataset, ABC):
    """
    All EEG datasets share the PyTorch Dataset protocol and a unified
    get_splits() interface.  Subclasses fill self.segments / self.labels
    and implement get_splits().
    """

    def __init__(self):
        self.segments = None  # np.ndarray (N, C, T)
        self.labels = None  # np.ndarray (N,)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        x = torch.tensor(self.segments[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    @abstractmethod
    def get_splits(self):
        """
        Return list of (train_indices, test_indices).
          - k-fold        →  k tuples
          - fixed / LOSO  →  1 or n_subjects tuples
        All indices are 1-D np.ndarray into this Dataset.
        """
        ...


# ══════════════════════════════════════════════════════════════════════════════
# 层一：SubjectDataset  —  SD（within-subject）
# ══════════════════════════════════════════════════════════════════════════════

class SubjectDatasetBase(EEGDatasetBase, ABC):
    """
    Single-subject dataset.  Subclasses additionally expose:
      self.trial_ids    np.ndarray (N,)  — optional
      self.session_ids  np.ndarray (N,)  — optional
    """

    def __init__(self):
        super().__init__()
        self.trial_ids = None
        self.session_ids = None


class DEAPSubjectDataset(SubjectDatasetBase):
    """
    cfg.split_by:
      "segment"  (default) — k-fold over all segments
      "trial"              — k-fold respecting trial boundaries
    """

    def __init__(self, cfg, subject_id):
        super().__init__()
        self.cfg = cfg
        segments, labels, trial_ids = load_deap_subject(cfg, subject_id)
        self.segments = segments
        self.labels = labels
        self.trial_ids = trial_ids

    def get_splits(self):
        split_by = getattr(self.cfg, "split_by", "segment")
        n_splits = self.cfg.n_splits
        seed = getattr(self.cfg, "seed", 42)
        if split_by == "trial":
            return kfold_by_unit(self.trial_ids, n_splits, random_state=seed)
        return kfold_by_segment(len(self), n_splits, random_state=seed)


class DREAMERSubjectDataset(SubjectDatasetBase):
    """
    cfg.split_by:
      "trial"   (default) — k-fold respecting trial boundaries
      "segment"           — k-fold over all segments
      "fixed"             — use cfg.train_trials / cfg.test_trials
    """

    def __init__(self, cfg, subject_id):
        super().__init__()
        self.cfg = cfg
        segments, labels, trial_ids = load_dreamer_subject(cfg, subject_id)
        self.segments = segments
        self.labels = labels
        self.trial_ids = trial_ids

    def get_splits(self):
        split_by = getattr(self.cfg, "split_by", "trial")
        n_splits = self.cfg.n_splits
        seed = getattr(self.cfg, "seed", 42)
        if split_by == "trial":
            return leave_one_trial_out(self.trial_ids)
        if split_by == "fixed":
            return fixed_split_by_unit(
                self.trial_ids, self.cfg.train_trials, self.cfg.test_trials
            )
        return kfold_by_segment(len(self), n_splits, random_state=seed)


class MAHNOBSubjectDataset(SubjectDatasetBase):
    """
    cfg.split_by:
      "session"  (default) — k-fold respecting session boundaries
      "segment"            — k-fold over all segments
      "fixed"              — use cfg.train_sessions / cfg.test_sessions
    """

    def __init__(self, cfg, subject_id):
        super().__init__()
        self.cfg = cfg
        segments, labels, session_ids = load_mahnob_subject(cfg, subject_id)
        if segments is None:
            # PASS subject — caller checks is_valid()
            self.segments = np.empty((0,), dtype=np.float32)
            self.labels = np.empty((0,), dtype=np.int64)
            self.session_ids = np.empty((0,), dtype=np.int64)
        else:
            self.segments = segments
            self.labels = labels
            self.session_ids = session_ids

    def is_valid(self):
        return len(self.segments) > 0

    def get_splits(self):
        split_by = getattr(self.cfg, "split_by", "session")
        n_splits = self.cfg.n_splits
        seed = getattr(self.cfg, "seed", 42)
        if split_by == "session":
            # return kfold_by_unit(self.session_ids, n_splits, random_state=seed)
            return leave_one_trial_out(self.session_ids)
        if split_by == "fixed":
            return fixed_split_by_unit(
                self.session_ids, self.cfg.train_sessions, self.cfg.test_sessions
            )
        return kfold_by_segment(len(self), n_splits, random_state=seed)


# ══════════════════════════════════════════════════════════════════════════════
# 层二：FullDataset  —  SI（LOSO / group k-fold）
# ══════════════════════════════════════════════════════════════════════════════

class FullDatasetBase(EEGDatasetBase, ABC):
    """
    Whole-dataset view: all subjects loaded at once.
    Exposes self.subject_ids so get_splits() can partition by subject.

    __getitem__ additionally returns subject_id as a third element,
    which is useful for domain-adaptation or contrastive objectives.
    Set _return_subject_id = False on a subclass to revert to (x, y).
    """
    _return_subject_id = True

    def __init__(self):
        super().__init__()
        self.subject_ids = None  # np.ndarray (N,)

    def __getitem__(self, idx):
        x = torch.tensor(self.segments[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        if self._return_subject_id:
            s = torch.tensor(self.subject_ids[idx], dtype=torch.long)
            return x, y, s
        return x, y


class DEAPFullDataset(FullDatasetBase):
    """
    cfg.si_split:
      "loso"         (default) — leave-one-subject-out
      "group_kfold"            — group k-fold over subjects (uses cfg.n_splits)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._load_all()

    def _load_all(self):
        all_segments, all_labels, all_subject_ids = [], [], []
        for sid in range(self.cfg.start_subject, self.cfg.num_subjects + 1):
            segs, lbls, _ = load_deap_subject(self.cfg, sid)
            if segs is None:
                continue
            all_segments.append(segs)
            all_labels.append(lbls)
            all_subject_ids.append(np.full(len(segs), sid, dtype=np.int64))
        self.segments = np.concatenate(all_segments)
        self.labels = np.concatenate(all_labels)
        self.subject_ids = np.concatenate(all_subject_ids)

    def get_splits(self):
        si_split = getattr(self.cfg, "si_split", "loso")
        seed = getattr(self.cfg, "seed", 42)
        if si_split == "group_kfold":
            return group_kfold_by_subject(
                self.subject_ids, n_splits=self.cfg.n_splits, random_state=seed
            )
        return leave_one_subject_out(self.subject_ids)


class DREAMERFullDataset(FullDatasetBase):
    """cfg.si_split: "loso" (default) | "group_kfold" """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._load_all()

    def _load_all(self):
        all_segments, all_labels, all_subject_ids = [], [], []
        for sid in range(self.cfg.start_subject, self.cfg.num_subjects + 1):
            segs, lbls, _ = load_dreamer_subject(self.cfg, sid)
            if segs is None:
                continue
            all_segments.append(segs)
            all_labels.append(lbls)
            all_subject_ids.append(np.full(len(segs), sid, dtype=np.int64))
        self.segments = np.concatenate(all_segments)
        self.labels = np.concatenate(all_labels)
        self.subject_ids = np.concatenate(all_subject_ids)

    def get_splits(self):
        si_split = getattr(self.cfg, "si_split", "loso")
        seed = getattr(self.cfg, "seed", 42)
        if si_split == "group_kfold":
            return group_kfold_by_subject(
                self.subject_ids, n_splits=self.cfg.n_splits, random_state=seed
            )
        return leave_one_subject_out(self.subject_ids)


class MAHNOBFullDataset(FullDatasetBase):
    """cfg.si_split: "loso" (default) | "group_kfold" """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._load_all()

    def _load_all(self):
        all_segments, all_labels, all_subject_ids = [], [], []
        for sid in range(self.cfg.start_subject, self.cfg.num_subjects + 1):
            segs, lbls, _ = load_mahnob_subject(self.cfg, sid)
            if segs is None:
                continue
            all_segments.append(segs)
            all_labels.append(lbls)
            all_subject_ids.append(np.full(len(segs), sid, dtype=np.int64))
        self.segments = np.concatenate(all_segments)
        self.labels = np.concatenate(all_labels)
        self.subject_ids = np.concatenate(all_subject_ids)

    def get_splits(self):
        si_split = getattr(self.cfg, "si_split", "loso")
        seed = getattr(self.cfg, "seed", 42)
        if si_split == "group_kfold":
            return group_kfold_by_subject(
                self.subject_ids, n_splits=self.cfg.n_splits, random_state=seed
            )
        return leave_one_subject_out(self.subject_ids)


# ══════════════════════════════════════════════════════════════════════════════
# 层三：MultiDatasetWrapper  —  跨数据集训练
# ══════════════════════════════════════════════════════════════════════════════

class MultiDatasetWrapper(EEGDatasetBase):
    """
    Wraps multiple FullDataset instances for cross-dataset training.

    Adds two extra fields per segment:
      self.subject_ids  — globally unique subject ID (offset per source dataset)
      self.dataset_ids  — which source dataset (0, 1, 2, ...)

    __getitem__ returns (x, y, dataset_id, subject_id).

    get_splits() strategy selected by cfg.multi_split:
      "loso"         — LOSO across the union of all subjects
      "group_kfold"  — group k-fold across subjects (uses cfg.n_splits)
      "by_dataset"   — leave-one-dataset-out; returns one split per source
                       dataset where that dataset is the test set
    """

    def __init__(self, full_datasets, cfg):
        super().__init__()
        self.cfg = cfg
        self._merge(full_datasets)

    def _merge(self, full_datasets):
        all_segments, all_labels = [], []
        all_dataset_ids, all_subject_ids = [], []
        subject_offset = 0

        for ds_idx, ds in enumerate(full_datasets):
            n = len(ds.segments)
            all_segments.append(ds.segments)
            all_labels.append(ds.labels)
            all_dataset_ids.append(np.full(n, ds_idx, dtype=np.int64))
            # offset subject IDs so they are globally unique across datasets
            shifted = ds.subject_ids + subject_offset
            all_subject_ids.append(shifted)
            subject_offset = int(shifted.max()) + 1

        self.segments = np.concatenate(all_segments)
        self.labels = np.concatenate(all_labels)
        self.dataset_ids = np.concatenate(all_dataset_ids)
        self.subject_ids = np.concatenate(all_subject_ids)

    def __getitem__(self, idx):
        x = torch.tensor(self.segments[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        ds = torch.tensor(self.dataset_ids[idx], dtype=torch.long)
        s = torch.tensor(self.subject_ids[idx], dtype=torch.long)
        return x, y, ds, s

    def get_splits(self):
        strategy = getattr(self.cfg, "multi_split", "loso")
        seed = getattr(self.cfg, "seed", 42)

        if strategy == "loso":
            return leave_one_subject_out(self.subject_ids)

        if strategy == "group_kfold":
            return group_kfold_by_subject(
                self.subject_ids, n_splits=self.cfg.n_splits, random_state=seed
            )

        if strategy == "by_dataset":
            # Leave-one-dataset-out: one split per source dataset
            unique_ds = np.unique(self.dataset_ids)
            splits = []
            for test_ds_id in unique_ds:
                train_units = unique_ds[unique_ds != test_ds_id]
                splits += fixed_split_by_unit(
                    self.dataset_ids,
                    train_units=train_units,
                    test_units=[test_ds_id],
                )
            return splits

        raise ValueError(f"Unknown multi_split strategy: {strategy!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 工厂函数
# ══════════════════════════════════════════════════════════════════════════════

_SUBJECT_DATASET_CLS = {
    "DEAP": DEAPSubjectDataset,
    "DREAMER": DREAMERSubjectDataset,
    "MAHNOB": MAHNOBSubjectDataset,
}

_FULL_DATASET_CLS = {
    "DEAP": DEAPFullDataset,
    "DREAMER": DREAMERFullDataset,
    "MAHNOB": MAHNOBFullDataset,
}


def build_subject_dataset(cfg, subject_id):
    """
    Instantiate a SubjectDataset for SD experiments.
    Returns None when the subject should be skipped (e.g. MAHNOB PASS).
    """
    name = cfg.dataset_name
    if name not in _SUBJECT_DATASET_CLS:
        raise ValueError(f"Unknown dataset: {name!r}")
    ds = _SUBJECT_DATASET_CLS[name](cfg, subject_id)
    if hasattr(ds, "is_valid") and not ds.is_valid():
        return None
    return ds


def build_full_dataset(cfg):
    """
    Instantiate a FullDataset (all subjects) for SI experiments.
    """
    name = cfg.dataset_name
    if name not in _FULL_DATASET_CLS:
        raise ValueError(f"Unknown dataset: {name!r}")
    return _FULL_DATASET_CLS[name](cfg)


def build_multi_dataset(cfgs):
    """
    Build a MultiDatasetWrapper from a list of per-dataset configs.
    The first cfg's multi_split / n_splits / seed settings govern splitting.

    cfgs : list of config objects, one per source dataset
    """
    full_datasets = [build_full_dataset(cfg) for cfg in cfgs]
    return MultiDatasetWrapper(full_datasets, cfg=cfgs[0])


# ══════════════════════════════════════════════════════════════════════════════
# 预训练 Dataset（保留原有，未改动）
# ══════════════════════════════════════════════════════════════════════════════

class EEGPretrainDataset(Dataset):
    def __init__(self, eeg, av_feat):
        self.eeg = eeg
        self.av = av_feat

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return self.eeg[idx], self.av[idx]
