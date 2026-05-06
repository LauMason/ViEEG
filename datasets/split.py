import numpy as np
from sklearn.model_selection import KFold, LeaveOneGroupOut


# ══════════════════════════════════════════════════════════════════════════════
# 策略一：所有 segment 打平后做 kfold
# 适用：SD，不需要感知 trial/session 边界（如 DEAP 默认模式）
# ══════════════════════════════════════════════════════════════════════════════

def kfold_by_segment(n_samples, n_splits, shuffle=True, random_state=42):
    """
    Standard k-fold directly over all segments.
    Returns list of (train_indices, test_indices), length == n_splits.
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return list(kf.split(np.arange(n_samples)))


# ══════════════════════════════════════════════════════════════════════════════
# 策略二：按结构单元（trial / session）做 kfold
# 适用：SD，保证同一 trial/session 的 segments 不跨越 train/test 边界
# ══════════════════════════════════════════════════════════════════════════════

def kfold_by_unit(unit_ids, n_splits, shuffle=True, random_state=42):
    """
    K-fold where fold boundaries respect trial/session boundaries.

    unit_ids : np.ndarray (N,) — trial_id or session_id of every segment.
    Returns list of (train_indices, test_indices), length == n_splits.
    """
    unique_units = np.unique(unit_ids)
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    splits = []
    for train_unit_pos, test_unit_pos in kf.split(unique_units):
        train_units = unique_units[train_unit_pos]
        test_units = unique_units[test_unit_pos]
        train_idx = np.where(np.isin(unit_ids, train_units))[0]
        test_idx = np.where(np.isin(unit_ids, test_units))[0]
        splits.append((train_idx, test_idx))

    return splits


def leave_one_trial_out(trial_ids):
    """
    每次拿一个 trial 作为 test，其余为 train
    """
    unique_trials = np.unique(trial_ids)

    splits = []

    for t in unique_trials:
        test_idx = np.where(trial_ids == t)[0]
        train_idx = np.where(trial_ids != t)[0]

        splits.append((train_idx, test_idx))

    return splits


# ══════════════════════════════════════════════════════════════════════════════
# 策略三：固定 train/test 单元
# 适用：SD/SI，数据集本身已规定哪些 trial/session/subject 为 train/test
# ══════════════════════════════════════════════════════════════════════════════

def fixed_split_by_unit(unit_ids, train_units, test_units):
    """
    Single predetermined split by unit membership.

    unit_ids    : np.ndarray (N,)
    train_units : array-like of unit IDs for the training set
    test_units  : array-like of unit IDs for the test set

    Returns list with exactly one (train_indices, test_indices) tuple.
    Kept as a list so all split strategies share the same interface.
    """
    train_idx = np.where(np.isin(unit_ids, train_units))[0]
    test_idx = np.where(np.isin(unit_ids, test_units))[0]
    return [(train_idx, test_idx)]


# ══════════════════════════════════════════════════════════════════════════════
# 策略四：按 subject 做 Leave-One-Subject-Out
# 适用：SI，FullDataset 整体加载后按 subject 边界划分
# ══════════════════════════════════════════════════════════════════════════════

def leave_one_subject_out(subject_ids):
    """
    Leave-one-subject-out cross-validation.
    Each fold: one subject is the test set, all others are training.

    subject_ids : np.ndarray (N,) — subject_id of every segment.
    Returns list of (train_indices, test_indices), length == n_unique_subjects.
    """
    logo = LeaveOneGroupOut()
    idx = np.arange(len(subject_ids))
    return [
        (train_idx, test_idx)
        for train_idx, test_idx in logo.split(idx, groups=subject_ids)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 策略五：按 subject 做 group k-fold
# 适用：SI，subject 数量多、不想做完整 LOSO 时
# ══════════════════════════════════════════════════════════════════════════════

def group_kfold_by_subject(subject_ids, n_splits, shuffle=True, random_state=42):
    """
    K-fold where each fold holds out one group of subjects.
    Equivalent to kfold_by_unit but named explicitly for subject-level use.

    subject_ids : np.ndarray (N,)
    Returns list of (train_indices, test_indices), length == n_splits.
    """
    return kfold_by_unit(
        subject_ids,
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
