"""
datasets/thingseeg.py  —  ViEEG 版（三路图像特征）

与 baseline 版的区别：
  get_image_data 返回三路特征：img / img_mask / img_mask01
  loader 返回 (eeg, img, img_mask, img_mask01) 四元组（训练/验证）
        或 (eeg, img, img_mask, img_mask01, label) 五元组（测试）
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


# ══════════════════════════════════════════════════════════════════════════════
# 底层 loader：SD
# ══════════════════════════════════════════════════════════════════════════════

def load_vieeg_subject(cfg, subject_id):
    """
    Load THINGS-EEG2 data for one subject with three image feature types.

    Returns:
        train_eeg       : (16540, 1, 63, 250)
        train_img       : (16540, D)   — original CLIP features
        train_img_mask  : (16540, D)   — masked image CLIP features
        train_img_mask01: (16540, D)   — mask01 image CLIP features
        test_eeg        : (200, 1, 63, 250)
        test_label      : (200,)  int64, 0..199
        test_img        : (200, D)
        test_img_mask   : (200, D)
        test_img_mask01 : (200, D)
    """
    sub_str = f"sub-{subject_id:02d}"
    eeg_root = cfg.eeg_data_path
    img_root = cfg.img_data_path
    clip_dir = getattr(cfg, "clip_dir", "CLIP-VIT-H-14")

    # ── EEG ──────────────────────────────────────────────────────────────
    train_eeg = np.load(
        os.path.join(eeg_root, sub_str, "preprocessed_eeg_training.npy"),
        allow_pickle=True,
    )["preprocessed_eeg_data"]
    train_eeg = np.expand_dims(np.mean(train_eeg, axis=1), axis=1)  # (16540,1,63,250)

    test_eeg = np.load(
        os.path.join(eeg_root, sub_str, "preprocessed_eeg_test.npy"),
        allow_pickle=True,
    )["preprocessed_eeg_data"]
    test_eeg = np.expand_dims(np.mean(test_eeg, axis=1), axis=1)  # (200,1,63,250)

    # ── Image features ───────────────────────────────────────────────────
    def _load(fname):
        return np.squeeze(np.load(
            os.path.join(img_root, clip_dir, fname), allow_pickle=True
        ))

    train_img = _load("clip_feature_maps_image_train.npy")  # (16540, D)
    test_img = _load("clip_feature_maps_image_test.npy")  # (200, D)
    train_img_mask = _load("clip_feature_maps_mask_image_train.npy")
    test_img_mask = _load("clip_feature_maps_mask_image_test.npy")
    train_img_mask01 = _load("clip_feature_maps_mask01_image_train.npy")
    test_img_mask01 = _load("clip_feature_maps_mask01_image_test.npy")

    test_label = np.arange(200, dtype=np.int64)

    print(f"ViEEG-SD {sub_str} | train: {train_eeg.shape} | test: {test_eeg.shape}")

    return (train_eeg, train_img, train_img_mask, train_img_mask01,
            test_eeg, test_label, test_img, test_img_mask, test_img_mask01)


# ══════════════════════════════════════════════════════════════════════════════
# 底层 loader：SI / LOSO
# ══════════════════════════════════════════════════════════════════════════════

def load_vieeg_subject_cross(cfg, subject_id):
    """
    Cross-subject version: other subjects → training set, target subject → test set.

    Image features are the same for all subjects and tiled once per other-subject.
    Returns same tuple as load_vieeg_subject.
    """
    eeg_root = cfg.eeg_data_path
    img_root = cfg.img_data_path
    clip_dir = getattr(cfg, "clip_dir", "CLIP-VIT-H-14")
    num_sub = cfg.num_subjects

    def _load(fname):
        return np.squeeze(np.load(
            os.path.join(img_root, clip_dir, fname), allow_pickle=True
        ))

    # shared image features
    tr_img = _load("clip_feature_maps_image_train.npy")  # (16540, D)
    te_img = _load("clip_feature_maps_image_test.npy")  # (200, D)
    tr_mask = _load("clip_feature_maps_mask_image_train.npy")
    te_mask = _load("clip_feature_maps_mask_image_test.npy")
    tr_mask01 = _load("clip_feature_maps_mask01_image_train.npy")
    te_mask01 = _load("clip_feature_maps_mask01_image_test.npy")

    # concat train + test image features for cross-subject training
    all_img = np.concatenate([tr_img, te_img], axis=0)  # (16740, D)
    all_mask = np.concatenate([tr_mask, te_mask], axis=0)
    all_mask01 = np.concatenate([tr_mask01, te_mask01], axis=0)

    train_eeg_list = []
    train_img_list, train_mask_list, train_mask01_list = [], [], []

    for sub in range(1, num_sub + 1):
        sub_str = f"sub-{sub:02d}"

        eeg_tr = np.load(
            os.path.join(eeg_root, sub_str, "preprocessed_eeg_training.npy"),
            allow_pickle=True,
        )["preprocessed_eeg_data"]
        eeg_tr = np.expand_dims(np.mean(eeg_tr, axis=1), axis=1)  # (16540,1,63,250)

        if sub == subject_id:
            # test subject: only use its test split for evaluation
            test_eeg = np.load(
                os.path.join(eeg_root, sub_str, "preprocessed_eeg_test.npy"),
                allow_pickle=True,
            )["preprocessed_eeg_data"]
            test_eeg = np.expand_dims(np.mean(test_eeg, axis=1), axis=1)  # (200,…)
        else:
            # other subjects: training set (train split only, matching original code)
            train_eeg_list.append(eeg_tr)
            train_img_list.append(tr_img)
            train_mask_list.append(tr_mask)
            train_mask01_list.append(tr_mask01)

    train_eeg = np.concatenate(train_eeg_list, axis=0)
    train_img = np.concatenate(train_img_list, axis=0)
    train_mask = np.concatenate(train_mask_list, axis=0)
    train_mask01 = np.concatenate(train_mask01_list, axis=0)

    test_label = np.arange(200, dtype=np.int64)

    print(f"ViEEG-SI (test={subject_id}) | train: {train_eeg.shape} | test: {test_eeg.shape}")

    return (train_eeg, train_img, train_mask, train_mask01,
            test_eeg, test_label, te_img, te_mask, te_mask01)


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Datasets
# ══════════════════════════════════════════════════════════════════════════════

class ViEEGTrainDataset(Dataset):
    """
    Training / validation dataset.
    Returns (eeg, img, img_mask, img_mask01) — 4-tuple.
    val_num rows (after shuffle) become the val split; the rest become train.
    """

    def __init__(self, eeg, img, img_mask, img_mask01,
                 split="train", val_num=740, seed=42):
        assert split in ("train", "val")

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(eeg))
        eeg = eeg[order]
        img = img[order]
        img_mask = img_mask[order]
        img_mask01 = img_mask01[order]

        if split == "val":
            sl = slice(None, val_num)
        else:
            sl = slice(val_num, None)

        self.eeg = torch.tensor(eeg[sl], dtype=torch.float32)
        self.img = torch.tensor(img[sl], dtype=torch.float32)
        self.img_mask = torch.tensor(img_mask[sl], dtype=torch.float32)
        self.img_mask01 = torch.tensor(img_mask01[sl], dtype=torch.float32)

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return self.eeg[idx], self.img[idx], self.img_mask[idx], self.img_mask01[idx]


class ViEEGTestDataset(Dataset):
    """
    Test dataset.
    Returns (eeg, img, img_mask, img_mask01, label) — 5-tuple.
    img / img_mask / img_mask01 are the per-class test image features,
    used for ablation evaluation inside the trainer.
    """

    def __init__(self, eeg, labels, img, img_mask, img_mask01):
        self.eeg = torch.tensor(eeg, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.img = torch.tensor(img, dtype=torch.float32)
        self.img_mask = torch.tensor(img_mask, dtype=torch.float32)
        self.img_mask01 = torch.tensor(img_mask01, dtype=torch.float32)

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return (self.eeg[idx], self.img[idx], self.img_mask[idx],
                self.img_mask01[idx], self.labels[idx])
