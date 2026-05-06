import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--device', default='cuda:0', type=str)
parser.add_argument('--image_dir', default='/data2/lmx/datasets/THINGS-EEG/Image/origin', type=str)
parser.add_argument('--mask_dir', default='/data2/lmx/datasets/THINGS-EEG/Image', type=str)
parser.add_argument('--segModel_dir', default='/data2/lmx/pretrained_models/birefnet', type=str)
parser.add_argument('--data_type', default='training_images', type=str)
# parser.add_argument('--data_type', default='test_images', type=str)


class SegmentationDataset(Dataset):
    def __init__(self, args, transform=None):
        self.image_dir = os.path.join(args.image_dir, args.data_type)
        self.mask_dir = args.mask_dir
        self.transform = transform
        self.image_paths = []

        # Traverse the directory to get all image paths
        for subdir in os.listdir(self.image_dir):
            subdir_path = os.path.join(self.image_dir, subdir)
            if os.path.isdir(subdir_path):
                image_files = [f for f in os.listdir(subdir_path) if f.endswith('.jpg')]
                for img_file in image_files:
                    self.image_paths.append(os.path.join(subdir_path, img_file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Load the image in RGB mode

        print(f"Processing image: {img_path}")  # Print the original image path
        print(f"InImage shape: {image.size}")  # 打印 PyTorch 张量的维度
        # Apply any transformations if provided
        if self.transform:
            image = self.transform(image)
        print(f"InImage shape: {image.size}")  # 打印 PyTorch 张量的维度

        # Generate the mask using DeepLabV3+ pre-trained model
        image, mask = self.generate_mask(image, img_path)
        return image, mask

    def generate_mask(self, image, img_path):
        # Load BiRefNet with weights
        from birefnet.birefnet import BiRefNet
        PATH_TO_WEIGHT = args.segModel_dir
        birefnet = BiRefNet.from_pretrained(PATH_TO_WEIGHT)
        torch.set_float32_matmul_precision(['high', 'highest'][0])
        device = args.device
        birefnet.to(device)
        birefnet.eval()

        transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_image = transform(image).unsqueeze(0).to(device)

        # Generate the mask prediction
        with torch.no_grad():
            # output = birefnet(input_image)['out'][0]
            output = birefnet(input_image)[-1].sigmoid().cpu()
        output = output[0].squeeze()
        pred_pil = transforms.ToPILImage()(output)
        mask = pred_pil.resize(image.size)
        image.putalpha(mask)
        self.save_mask(image, mask, img_path)

        mask_np = np.array(mask)  # Convert image from PIL Image to numpy array
        image_np = np.array(image)  # Convert image from PIL Image to numpy array
        return torch.tensor(image_np, dtype=torch.long), torch.tensor(mask_np, dtype=torch.long)

    def save_mask(self, image, mask, img_path):
        try:
            # Create the directory structure for saving masks
            mask_dir = os.path.join(self.mask_dir, 'mask_01', args.data_type,
                                    os.path.relpath(os.path.dirname(img_path), self.image_dir))
            image_dir = os.path.join(self.mask_dir, 'mask', args.data_type,
                                     os.path.relpath(os.path.dirname(img_path), self.image_dir))
            os.makedirs(mask_dir, exist_ok=True)
            os.makedirs(image_dir, exist_ok=True)

            # Generate the mask file path
            filename = os.path.basename(img_path).replace('.jpg', '_mask.png')
            mask_path = os.path.join(mask_dir, filename)
            image_path = os.path.join(image_dir, filename)

            print(f"MImage shape: {image.size}")  # 打印 PyTorch 张量的维度
            print(f"M01Image shape: {mask.size}")  # 打印 PyTorch 张量的维度

            # Save mask as an image
            mask.save(mask_path)

            # Save image as an image
            image.save(image_path)

            print(f"Saved mask to: {mask_path}")

        except Exception as e:
            print(f"Error saving mask for {img_path}: {e}")


if __name__ == '__main__':
    args = parser.parse_args()
    print('\nInput arguments:')
    for key, val in vars(args).items():
        print('{:16} {}'.format(key, val))

    # Define the dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        # transforms.ToTensor(),
    ])

    dataset = SegmentationDataset(args, transform=transform)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Training loop (example)
    for images, masks in train_loader:
        # Process images and masks here
        pass
