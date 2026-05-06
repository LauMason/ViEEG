DATASET_INFO = {

    "THINGS-EEG": {
        # EEG 采样率（250 Hz 预处理版本）
        "sampling_rate": 250,
        # 电极数（Things-EEG2 使用 63 通道）
        "num_electrodes": 63,
        # 时间点数（250 Hz × 1 s 刺激窗口）
        "time_points": 250,
        # 被试数
        "num_subjects": 10,
        # 训练集图像类别数 / 测试集图像类别数
        "num_train_concepts": 16540,
        "num_classes": 200,          # test set: 200 concepts，用于 Top-K 检索
        # 任务类型
        "task_type": "retrieval",    # 区别于情绪数据集的 classification
        # chunk_size / overlap 对脑解码不适用，置 None
        "chunk_size": None,
        "overlap": None,
        "threshold": None,
    },
}