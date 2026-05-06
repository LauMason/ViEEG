import os
import json
import pandas as pd
from datetime import datetime
import yaml


class ExperimentLogger:

    def __init__(self, cfg, mode="WithinSubject"):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{mode}_{timestamp}"
        self.base_dir = os.path.join(cfg.save_dir, exp_name)
        os.makedirs(self.base_dir, exist_ok=True)

        self.model_dir = os.path.join(self.base_dir, 'models')
        self.log_dir = os.path.join(self.base_dir, 'logs')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.cfg = cfg
        self.mode = mode

        # 保存 config
        self.subject_summary_records = []
        self.save_config()
        self.subject_records = []

    # =============================
    # Save config
    # =============================
    def save_config(self):

        config_dict = {
            k: getattr(self.cfg, k)
            for k in dir(self.cfg)
            if not k.startswith("__") and not callable(getattr(self.cfg, k))
        }

        with open(os.path.join(self.base_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

    # =============================
    # Model directory
    # =============================
    def get_model_dir(self, subject, fold=None):

        if fold is None:
            save_dir = os.path.join(
                self.model_dir,
                f"subject_{subject}"
            )
        else:
            save_dir = os.path.join(
                self.model_dir,
                f"subject_{subject}",
                f"fold_{fold}"
            )

        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def log_subject_result(self, subject, best_acc, best_epoch, history, fold=None):

        # ---------- 1. 构建文件名 ----------
        if fold is not None:
            history_name = f"subject_{subject}_fold_{fold}_history.csv"
        else:
            history_name = f"subject_{subject}_history.csv"

        subject_path = os.path.join(self.log_dir, f"subject_{subject}")
        os.makedirs(subject_path, exist_ok=True)
        history_path = os.path.join(subject_path, history_name)

        # ---------- 2. 保存 epoch 级记录 ----------
        df = pd.DataFrame(history)
        df.to_csv(history_path, index=False)

        # ---------- 3. 构建 summary 记录 ----------
        record = {
            "subject": subject,
            "best_acc": best_acc,
            "best_epoch": best_epoch
        }

        # 只有 fold 不为 None 才加入 fold 字段
        if fold is not None:
            record["fold"] = fold

        self.subject_records.append(record)

    # =============================
    # Log subject best results
    # =============================
    def log_subject_best(self, subject, fold_best_results):
        """
        fold_best_results = [
            {"fold":1, "best_acc":0.75, "best_epoch":23},
            ...
        ]
        """

        subject_dir = os.path.join(self.log_dir, f"subject_{subject}")
        os.makedirs(subject_dir, exist_ok=True)

        df = pd.DataFrame(fold_best_results)
        df.to_csv(os.path.join(subject_dir, "subject_best_summary.csv"), index=False)

        mean_acc = df["best_acc"].mean()
        std_acc = df["best_acc"].std()

        record = {
            "subject": subject,
            "mean_acc": mean_acc,
            "std_acc": std_acc
        }

        self.subject_summary_records.append(record)

        # ===== 实时写入 overall_summary.csv =====
        overall_path = os.path.join(self.base_dir, "overall_summary.csv")

        if os.path.exists(overall_path):
            df_old = pd.read_csv(overall_path)
            df_old = df_old[df_old["subject"] != "OVERALL"]
            df_new = pd.concat([df_old, pd.DataFrame([record])], ignore_index=True)
        else:
            df_new = pd.DataFrame([record])

        df_new.to_csv(overall_path, index=False)

        print(f"Subject {subject} Mean: {mean_acc:.4f} ± {std_acc:.4f}")

    # =============================
    # Log overall
    # =============================
    def log_overall(self):

        overall_path = os.path.join(self.base_dir, "overall_summary.csv")

        df = pd.read_csv(overall_path)

        overall_mean = df["mean_acc"].mean()
        overall_std = df["mean_acc"].std()

        overall_row = pd.DataFrame([{
            "subject": "OVERALL",
            "mean_acc": overall_mean,
            "std_acc": overall_std
        }])

        df = pd.concat([df, overall_row], ignore_index=True)
        df.to_csv(overall_path, index=False)

        print("\n" + "=" * 60)
        print(f"Overall: {overall_mean:.4f} ± {overall_std:.4f}")
        print("=" * 60)

    def log_subject_best_top5(self, subject, fold_best_results):
        """
        fold_best_results = [
            {"fold":1, "best_acc":0.75, "best_epoch":23},
            ...
        ]
        """

        subject_dir = os.path.join(self.log_dir, f"subject_{subject}")
        os.makedirs(subject_dir, exist_ok=True)

        df = pd.DataFrame(fold_best_results)
        df.to_csv(os.path.join(subject_dir, "subject_best_summary.csv"), index=False)

        mean_acc = df["best_acc"].mean()
        std_acc = df["best_acc"].std()
        mean_acc3 = df["top3"].mean()
        std_acc3 = df["top3"].std()
        mean_acc5 = df["top5"].mean()
        std_acc5 = df["top5"].std()

        record = {
            "subject": subject,
            "mean_acc_top1": mean_acc,
            "std_acc_top1": std_acc,
            "mean_acc_top3": mean_acc3,
            "std_acc_top3": std_acc3,
            "mean_acc_top5": mean_acc5,
            "std_acc_top5": std_acc5
        }

        self.subject_summary_records.append(record)

        # ===== 实时写入 overall_summary.csv =====
        overall_path = os.path.join(self.base_dir, "overall_summary.csv")

        if os.path.exists(overall_path):
            df_old = pd.read_csv(overall_path)
            df_old = df_old[df_old["subject"] != "OVERALL"]
            df_new = pd.concat([df_old, pd.DataFrame([record])], ignore_index=True)
        else:
            df_new = pd.DataFrame([record])

        df_new.to_csv(overall_path, index=False)

        print(f"Subject {subject} Mean Top-1 ACC: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Subject {subject} Mean Top-3 ACC: {mean_acc3:.4f} ± {std_acc3:.4f}")
        print(f"Subject {subject} Mean Top-5 ACC: {mean_acc5:.4f} ± {std_acc5:.4f}")

    # =============================
    # Log overall
    # =============================
    def log_overall_top5(self):

        overall_path = os.path.join(self.base_dir, "overall_summary.csv")

        df = pd.read_csv(overall_path)

        overall_mean = df["mean_acc_top1"].mean()
        overall_std = df["mean_acc_top1"].std()
        overall_mean3 = df["mean_acc_top3"].mean()
        overall_std3 = df["mean_acc_top3"].std()
        overall_mean5 = df["mean_acc_top5"].mean()
        overall_std5 = df["mean_acc_top5"].std()

        overall_row = pd.DataFrame([{
            "subject": "OVERALL",
            "mean_acc_top1": overall_mean,
            "std_acc_top1": overall_std,
            "mean_acc_top3": overall_mean3,
            "std_acc_top3": overall_std3,
            "mean_acc_top5": overall_mean5,
            "std_acc_top5": overall_std5,
        }])

        df = pd.concat([df, overall_row], ignore_index=True)
        df.to_csv(overall_path, index=False)

        print("\n" + "=" * 60)
        print(f"Overall Top-1 ACC: {overall_mean:.4f} ± {overall_std:.4f}")
        print(f"Overall Top-3 ACC: {overall_mean3:.4f} ± {overall_std3:.4f}")
        print(f"Overall Top-5 ACC: {overall_mean5:.4f} ± {overall_std5:.4f}")
        print("=" * 60)
