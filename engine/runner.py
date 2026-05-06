"""
engine/vieeg_runner.py

ViEEG experiment Runner（三路图像特征版）。

与 bd_runner.py（baseline）的区别：
  - 使用 load_vieeg_subject / load_vieeg_subject_cross 加载三路图像特征
  - Dataset 是 ViEEGTrainDataset / ViEEGTestDataset
  - 调用 ViEEGTrainer.fit()，记录主指标 + ablation
  - 日志用 saver.log_subject_best_top5 + log_overall_top5
"""

import copy
from torch.utils.data import DataLoader

from datasets.thingseeg_vieeg import (
    load_vieeg_subject,
    load_vieeg_subject_cross,
    ViEEGTrainDataset,
    ViEEGTestDataset,
)
from engine.trainer import ViEEGTrainer
from models.build import build_model
from utils.saver import ExperimentLogger


class ViEEGExperiment:

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.model = build_model(cfg).to(self.device)
        self._init_state = copy.deepcopy(self.model.state_dict())

    def reset_model(self):
        self.model.load_state_dict(self._init_state)

    # ── 入口 ──────────────────────────────────────────────────────────────

    def run(self):
        if self.cfg.exp_setting == "SD":
            self._subject_dependent()
        elif self.cfg.exp_setting == "SI":
            self._subject_independent()
        else:
            raise ValueError(
                f"Unknown exp_setting: {self.cfg.exp_setting!r}. "
                "Expected 'SD' or 'SI'."
            )

    # ── SD ────────────────────────────────────────────────────────────────

    def _subject_dependent(self):
        logger = ExperimentLogger(self.cfg, mode="ViEEG_WithinSubject")

        for subject in range(self.cfg.start_subject, self.cfg.num_subjects + 1):
            print(f"\n===== Subject {subject} =====")

            data = load_vieeg_subject(self.cfg, subject)
            (train_eeg, train_img, train_mask, train_mask01,
             test_eeg, test_label, test_img, test_mask, test_mask01) = data

            subject_scores = []

            for fold_idx in range(self.cfg.fold):
                fold_seed = self.cfg.seed + fold_idx
                print(f"\n--- Fold {fold_idx + 1} / {self.cfg.fold}  (seed={fold_seed}) ---")

                self.reset_model()
                save_dir = logger.get_model_dir(subject, fold_idx + 1)
                trainer = ViEEGTrainer(self.cfg, self.model, save_dir=save_dir)

                train_loader, val_loader, test_loader = self._build_loaders(
                    train_eeg, train_img, train_mask, train_mask01,
                    test_eeg, test_label, test_img, test_mask, test_mask01,
                    seed=fold_seed,
                )

                history, top1, top3, top5, best_epoch, ablation = \
                    trainer.fit(train_loader, val_loader, test_loader)

                logger.log_subject_result(
                    subject=subject,
                    fold=fold_idx + 1,
                    best_acc=top1,
                    best_epoch=best_epoch,
                    history=history,
                )
                score = {
                    "fold": fold_idx + 1,
                    "best_acc": top1,
                    "best_epoch": best_epoch,
                    "top3": top3,
                    "top5": top5,
                }
                # add ablation to fold record
                if ablation:
                    for name, res in ablation.items():
                        key = name.replace("+", "_")
                        score[f"abl_{key}_top1"] = res[1]
                        score[f"abl_{key}_top3"] = res[3]
                        score[f"abl_{key}_top5"] = res[5]
                subject_scores.append(score)

                self._print_ablation(subject, fold_idx + 1, ablation)

            logger.log_subject_best_top5(subject, subject_scores)

        logger.log_overall_top5()

    # ── SI / LOSO ─────────────────────────────────────────────────────────

    def _subject_independent(self):
        logger = ExperimentLogger(self.cfg, mode="ViEEG_LOSO")

        for subject in range(self.cfg.start_subject, self.cfg.num_subjects + 1):
            print(f"\n=== Leave Subject {subject} Out ===")

            data = load_vieeg_subject_cross(self.cfg, subject)
            (train_eeg, train_img, train_mask, train_mask01,
             test_eeg, test_label, test_img, test_mask, test_mask01) = data

            subject_scores = []

            for fold_idx in range(self.cfg.fold):
                fold_seed = self.cfg.seed + fold_idx
                print(f"\n--- Fold {fold_idx + 1} / {self.cfg.fold}  (seed={fold_seed}) ---")

                self.reset_model()
                save_dir = logger.get_model_dir(subject, fold_idx + 1)
                trainer = ViEEGTrainer(self.cfg, self.model, save_dir=save_dir)

                train_loader, val_loader, test_loader = self._build_loaders(
                    train_eeg, train_img, train_mask, train_mask01,
                    test_eeg, test_label, test_img, test_mask, test_mask01,
                    seed=fold_seed,
                )

                history, top1, top3, top5, best_epoch, ablation = \
                    trainer.fit(train_loader, val_loader, test_loader)

                logger.log_subject_result(
                    subject=subject,
                    fold=fold_idx + 1,
                    best_acc=top1,
                    best_epoch=best_epoch,
                    history=history,
                )
                score = {
                    "fold": fold_idx + 1,
                    "best_acc": top1,
                    "best_epoch": best_epoch,
                    "top3": top3,
                    "top5": top5,
                }
                if ablation:
                    for name, res in ablation.items():
                        key = name.replace("+", "_")
                        score[f"abl_{key}_top1"] = res[1]
                        score[f"abl_{key}_top3"] = res[3]
                        score[f"abl_{key}_top5"] = res[5]
                subject_scores.append(score)

                self._print_ablation(subject, fold_idx + 1, ablation)

            logger.log_subject_best_top5(subject, subject_scores)

        logger.log_overall_top5()

    # ── 内部工具 ──────────────────────────────────────────────────────────

    def _build_loaders(self,
                       train_eeg, train_img, train_mask, train_mask01,
                       test_eeg, test_label, test_img, test_mask, test_mask01,
                       seed):
        val_num = getattr(self.cfg, "val_num", 740)

        train_ds = ViEEGTrainDataset(
            train_eeg, train_img, train_mask, train_mask01,
            split="train", val_num=val_num, seed=seed,
        )
        val_ds = ViEEGTrainDataset(
            train_eeg, train_img, train_mask, train_mask01,
            split="val", val_num=val_num, seed=seed,
        )
        test_ds = ViEEGTestDataset(
            test_eeg, test_label, test_img, test_mask, test_mask01,
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size_train,
            shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size_train,
            shuffle=False, num_workers=0,
        )
        test_loader = DataLoader(
            test_ds, batch_size=self.cfg.batch_size_test,
            shuffle=False, num_workers=0,
        )
        return train_loader, val_loader, test_loader

    @staticmethod
    def _print_ablation(subject, fold, ablation):
        if not ablation:
            return
        print(f"\n  [Subject {subject} / Fold {fold}] Ablation results:")
        for name, res in ablation.items():
            print(f"    {name:16s}  Top1:{res[1]:.4f}  Top3:{res[3]:.4f}  Top5:{res[5]:.4f}")
