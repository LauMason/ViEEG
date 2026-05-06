from models.eegnet import EEGNet
from models.demo import Demo
from models.comparisons.astdfnet import ASTDFNET
from models.VISTA.VISTA import VISTA
from models.comparisons.NICE import NICE
from models.ViEEG.ViEEG import ViEEG


def build_model(cfg):

    if cfg.model_name == "Demo":
        return Demo(cfg, num_classes=2)
    if cfg.model_name == "EEGNet":
        return EEGNet(cfg, num_classes=2)
    elif cfg.model_name == "VISTA":
        return VISTA()
    elif cfg.model_name == "ASTDF-NET":
        return ASTDFNET()
    elif cfg.model_name == "NICE":
        return NICE()
    elif cfg.model_name == "ViEEG":
        return ViEEG()
    else:
        raise ValueError("Unknown model")
