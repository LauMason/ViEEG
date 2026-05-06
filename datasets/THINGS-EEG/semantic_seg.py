import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_paths = []

        # Traverse the directory to get all image paths
        for subdir in os.listdir(image_dir):
            subdir_path = os.path.join(image_dir, subdir)
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

        # Apply any transformations if provided
        if self.transform:
            image = self.transform(image)
        print(f"Image shape: {image.shape}")  # 打印 PyTorch 张量的维度

        # Generate the mask using DeepLabV3+ pre-trained model
        mask = self.generate_mask(image, img_path)

        return image, mask

    def generate_mask(self, image, img_path):
        # Load the pre-trained DeepLabV3 model with ResNet-101 backbone
        deeplab_model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        fcn_model = models.segmentation.fcn_resnet101(pretrained=True)

        entrypoints = torch.hub.list("pytorch/vision", force_reload=True)
        print(entrypoints)
        model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)

        mask_model = model
        mask_model.eval()  # Set to evaluation mode

        # Move model to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask_model.to(device)

        # If the image is already a tensor, no need to apply transform
        # if isinstance(image, torch.Tensor):
        #     input_image = image.unsqueeze(0).to(device)
        # else:
        # Preprocess the input image for model input
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_image = transform(image).unsqueeze(0).to(device)
        print(f"1Image shape: {image.shape}")  # 打印 PyTorch 张量的维度

        # Generate the mask prediction
        with torch.no_grad():
            output = mask_model(input_image)['out'][0]

        # Convert output to binary mask (considering background vs. object)
        output_predictions = output.argmax(0)  # Class with the highest score

        # Create a mask where foreground is 1 and background is 0
        mask = output_predictions.cpu().numpy()
        print(f"MImage shape: {image.shape}")  # 打印 PyTorch 张量的维度

        # Save the generated mask to the corresponding directory structure
        self.save_mask(mask, img_path)

        return torch.tensor(mask, dtype=torch.long)

    def save_mask(self, mask, img_path):
        # Create the directory structure for saving masks
        mask_dir = os.path.join(self.mask_dir, os.path.relpath(os.path.dirname(img_path), self.image_dir))
        os.makedirs(mask_dir, exist_ok=True)

        # Generate the mask file path
        mask_filename = os.path.basename(img_path).replace('.jpg', '_mask.png')
        mask_path = os.path.join(mask_dir, mask_filename)

        # Save mask as an image
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255)  # Convert to 0 or 255 (black/white)
        mask_image.save(mask_path)

        print(f"Saved mask to: {mask_path}")


# Example usage
image_dir = '/lab_202/2023/lmx/Data/Things_EEG2/Image/origin/training_images'  # Replace with your image directory
mask_dir = '/lab_202/2023/lmx/Data/Things_EEG2/Image/masks'  # Directory where masks will be saved

# Define the dataset and dataloader
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Training loop (example)
for images, masks in train_loader:
    # Process images and masks here
    pass


