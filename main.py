from utils.seed import set_seed
from config.baseline_config import BrainDecodingConfig
from engine.baseline_runner import BrainDecodingExperiment
from config.main_config import ViEEGConfig
from engine.runner import ViEEGExperiment

mode = 'baseline'  # 'baseline' | 'vieeg'


def main():
    if mode == 'baseline':
        my_config = BrainDecodingConfig
        my_experiment = BrainDecodingExperiment
    elif mode == 'vieeg':
        my_config = ViEEGConfig
        my_experiment = ViEEGExperiment
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    cfg = my_config()
    set_seed(cfg.seed)
    exp = my_experiment(cfg)
    exp.run()


if __name__ == "__main__":
    main()
