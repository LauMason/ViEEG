import os
from datasets.dataset_info import DATASET_INFO


class ViEEGConfig:

    def __init__(self):
        # ========== 数据集 ==========
        self.dataset_name = "THINGS-EEG"
        self.exp_setting = "SD"  # "SD" | "SI"
        self.start_subject = 1

        # ── 数据路径 ─────────────────────────────────────────────────────
        self.eeg_data_path = "/data2/lmx/datasets/THINGS-EEG/Preprocessed_data_250Hz"
        self.img_data_path = "/data2/lmx/datasets/THINGS-EEG/EMBEDDING"
        # CLIP 特征子目录（存放 img / mask / mask01 三组 npy 文件）
        self.clip_dir = "CLIP-VIT-H-14"

        # ── 数据划分 ─────────────────────────────────────────────────────
        self.val_num = 740  # 从 train split 切出多少条作验证集

        # ========== 训练超参数 ==========
        self.fold = 5  # 每个 subject 重复训练次数（不同 seed）
        self.batch_size_train = 1000
        self.batch_size_test = 400
        self.max_epochs = 80  # SD 推荐 80；SI 推荐 40
        self.lr = 2e-4
        self.weight_decay = 0.0
        self.optimizer = "Adam"
        self.scheduler = "cosine"

        # ========== 模型 ==========
        self.model_name = "ViEEG"  # 在 models/build.py 中注册
        self.n_layer = 1  # ViEEG transformer layers
        self.n_head = 1  # ViEEG attention heads

        # ========== 公共 ==========
        self.seed = 2023
        self.device = "cuda"
        self.devices = 1

        # ========== 保存 ==========
        self.save_model = True
        self.save_logger = True

        # ── 从 DATASET_INFO 自动填充 ──────────────────────────────────────
        self._apply_dataset_info()

    def _apply_dataset_info(self):
        info = DATASET_INFO.get(self.dataset_name)
        if info is None:
            raise ValueError(
                f"Unknown dataset: {self.dataset_name!r}. "
                f"Available: {list(DATASET_INFO.keys())}"
            )
        self.num_subjects = info["num_subjects"]
        self.num_electrodes = info["num_electrodes"]
        self.sampling_rate = info.get("sampling_rate")
        self.num_classes = info.get("num_classes", 200)
        self.dataset_info = info

        self.save_dir = os.path.join(
            "./experiments/brain_decoding",
            self.dataset_name, self.exp_setting, self.model_name,
        )
