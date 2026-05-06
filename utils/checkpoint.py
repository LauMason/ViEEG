import torch
import os


def save_checkpoint(model, save_dir, filename):

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), path)
