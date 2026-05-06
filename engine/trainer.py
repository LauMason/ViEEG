"""
engine/vieeg_trainer.py

ViEEG Trainer：三路 EEG ↔ 图像对比学习。

与 BrainDecodingTrainer（baseline）的核心差异：
  - model 输出三个特征 (v_feat, m_feat, b_feat)
  - 每个 batch 输入 (eeg, img, img_mask, img_mask01)
  - loss = 单次 InfoNCE，输入是三路 cat 后的特征
  - test 时同时计算主指标（三路 cat）和 6 种 ablation 的 Top-K
"""

import numpy as np
import torch
import torch.nn as nn
from utils.metrics_vieeg import topk_retrieval_accuracy, vieeg_ablation_topk
from utils.checkpoint import save_checkpoint


class ViEEGTrainer:

    def __init__(self, cfg, model, save_dir=None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)
        self.save_dir = save_dir
        self.save_model = cfg.save_model

        self.criterion = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(
            torch.ones([], device=self.device) * np.log(1 / 0.07)
        )

        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.max_epochs
        )

        self.best_val_loss = np.inf
        self.best_epoch = 0
        self.best_top1 = 0.0
        self.best_top3 = 0.0
        self.best_top5 = 0.0

    def _build_optimizer(self):
        params = list(self.model.parameters()) + [self.logit_scale]
        return torch.optim.Adam(
            params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=(0.5, 0.999),
        )

    # ── 三路对比前向 ──────────────────────────────────────────────────────

    def _contrastive_step(self, eeg, img, img_mask, img_mask01):
        """
        Forward + symmetric InfoNCE on concatenated 3-branch features.
        Returns (loss, v_feat_norm, m_feat_norm, b_feat_norm).
        """
        label = torch.arange(eeg.shape[0], device=self.device)

        v_feat, m_feat, b_feat = self.model(eeg)  # each (B, proj_dim)

        # L2 normalise each branch
        v_feat = v_feat / v_feat.norm(dim=1, keepdim=True)
        m_feat = m_feat / m_feat.norm(dim=1, keepdim=True)
        b_feat = b_feat / b_feat.norm(dim=1, keepdim=True)
        img = img / img.norm(dim=1, keepdim=True)
        img_mask = img_mask / img_mask.norm(dim=1, keepdim=True)
        img_mask01 = img_mask01 / img_mask01.norm(dim=1, keepdim=True)

        # concatenate all three branches → single joint embedding
        eeg_joint = torch.cat([v_feat, m_feat, b_feat], dim=1)  # (B, 3D)
        img_joint = torch.cat([img, img_mask, img_mask01], dim=1)

        # re-normalise the joint vector
        eeg_joint = eeg_joint / eeg_joint.norm(dim=1, keepdim=True)
        img_joint = img_joint / img_joint.norm(dim=1, keepdim=True)

        scale = self.logit_scale.exp()
        logits = scale * eeg_joint @ img_joint.t()  # (B, B)
        loss = self.criterion(logits, label)

        return loss, v_feat, m_feat, b_feat

    # ── Train ─────────────────────────────────────────────────────────────

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for eeg, img, img_mask, img_mask01 in loader:
            eeg = eeg.to(self.device)
            img = img.to(self.device)
            img_mask = img_mask.to(self.device)
            img_mask01 = img_mask01.to(self.device)

            self.optimizer.zero_grad()
            loss, *_ = self._contrastive_step(eeg, img, img_mask, img_mask01)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    # ── Validate ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0

        for eeg, img, img_mask, img_mask01 in loader:
            eeg = eeg.to(self.device)
            img = img.to(self.device)
            img_mask = img_mask.to(self.device)
            img_mask01 = img_mask01.to(self.device)
            loss, *_ = self._contrastive_step(eeg, img, img_mask, img_mask01)
            total_loss += loss.item()

        return total_loss / len(loader)

    # ── Test ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def test(self, loader):
        """
        loader yields (eeg, img, img_mask, img_mask01, label) — 5-tuple.

        Returns:
            main_topk   : dict {1: top1, 3: top3, 5: top5}  — 三路 cat 主指标
            ablation    : dict of 6 ablation results (from vieeg_ablation_topk)
        """
        self.model.eval()

        all_v, all_m, all_b = [], [], []
        all_vi, all_mi, all_bi = [], [], []
        all_labels = []

        for eeg, img, img_mask, img_mask01, label in loader:
            eeg = eeg.to(self.device)
            img = img.to(self.device)
            img_mask = img_mask.to(self.device)
            img_mask01 = img_mask01.to(self.device)

            v_feat, m_feat, b_feat = self.model(eeg)
            v_feat = v_feat / v_feat.norm(dim=1, keepdim=True)
            m_feat = m_feat / m_feat.norm(dim=1, keepdim=True)
            b_feat = b_feat / b_feat.norm(dim=1, keepdim=True)
            img = img / img.norm(dim=1, keepdim=True)
            img_mask = img_mask / img_mask.norm(dim=1, keepdim=True)
            img_mask01 = img_mask01 / img_mask01.norm(dim=1, keepdim=True)

            all_v.append(v_feat);
            all_vi.append(img)
            all_m.append(m_feat);
            all_mi.append(img_mask)
            all_b.append(b_feat);
            all_bi.append(img_mask01)
            all_labels.append(label.to(self.device))

        v_eeg = torch.cat(all_v);
        v_img = torch.cat(all_vi)
        m_eeg = torch.cat(all_m);
        m_img = torch.cat(all_mi)
        b_eeg = torch.cat(all_b);
        b_img = torch.cat(all_bi)
        labels = torch.cat(all_labels)

        # ── main metric: three-branch cat ────────────────────────────────
        eeg_joint = torch.cat([v_eeg, m_eeg, b_eeg], dim=1)
        img_joint = torch.cat([v_img, m_img, b_img], dim=1)
        eeg_joint = eeg_joint / eeg_joint.norm(dim=1, keepdim=True)
        img_joint = img_joint / img_joint.norm(dim=1, keepdim=True)
        main_topk = topk_retrieval_accuracy(eeg_joint, img_joint, labels, ks=(1, 3, 5))

        # ── ablation ─────────────────────────────────────────────────────
        ablation = vieeg_ablation_topk(
            v_eeg, m_eeg, b_eeg,
            v_img, m_img, b_img,
            labels, ks=(1, 3, 5),
        )

        return main_topk, ablation

    # ── Main loop ─────────────────────────────────────────────────────────

    def fit(self, train_loader, val_loader, test_loader):
        """
        Returns (history, best_top1, best_top3, best_top5, best_epoch, best_ablation)
        best_ablation is the ablation dict at the best-val-loss epoch.
        """
        n_epochs = self.cfg.max_epochs
        history = []
        best_ablation = None

        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss = self.validate(val_loader)
            main_topk, ablation = self.test(test_loader)

            self.scheduler.step()

            top1, top3, top5 = main_topk[1], main_topk[3], main_topk[5]
            print(
                f"Epoch [{epoch}/{n_epochs}] "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Top1: {top1:.4f}  Top3: {top3:.4f}  Top5: {top5:.4f}"
            )

            # print ablation summary
            for name, res in ablation.items():
                print(f"  [{name}] Top1:{res[1]:.4f} Top3:{res[3]:.4f} Top5:{res[5]:.4f}")

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "top1": top1, "top3": top3, "top5": top5,
            }
            # flatten ablation into history row
            for name, res in ablation.items():
                key = name.replace("+", "_")
                row[f"abl_{key}_top1"] = res[1]
                row[f"abl_{key}_top3"] = res[3]
                row[f"abl_{key}_top5"] = res[5]
            history.append(row)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_top1 = top1
                self.best_top3 = top3
                self.best_top5 = top5
                best_ablation = ablation
                if self.save_model:
                    save_checkpoint(self.model, self.save_dir, "best_vieeg_model.pth")

        return history, self.best_top1, self.best_top3, self.best_top5, self.best_epoch, best_ablation
