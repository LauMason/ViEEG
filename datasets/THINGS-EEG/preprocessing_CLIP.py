"""
PCA is performed on the DNN feature maps to reduce their dimensionality.
PCA is applied on either the feature maps of single DNN layers, or on the
appended feature maps of all layers.

Parameters
----------
dnn : str
    Used DNN among 'clip_vit_large_patch14'.
pretrained : bool
    If True, use the pretrained CLIP feature maps.
layers : str
    Whether to use 'all' or 'single' layers.
n_components : int
    Number of DNN feature maps PCA components retained.
project_dir : str
    Directory of the project folder.

"""

import argparse
import numpy as np
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
import traceback

# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', default='/data2/lmx/datasets/THINGS-EEG/EMBEDDING', type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layers', default='single', type=str)
parser.add_argument('--n_components', default=3000, type=int)
parser.add_argument('--project_dir0', default='LAION', type=str)
parser.add_argument('--project_dir', default='CLIP-ViT-B-32', type=str)
parser.add_argument('--device', default='cuda:0', type=str)
args = parser.parse_args()

print('>>> Apply PCA on the feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

# Load CLIP model and processor
base_path = '/data2/lmx/'
clip_path = os.path.join(base_path, 'pretrained_models', args.project_dir0, args.project_dir)
model = CLIPModel.from_pretrained(clip_path)
processor = CLIPProcessor.from_pretrained(clip_path)
model.eval()  # Ensure the model is in evaluation mode
device = args.device
model.to(device)


# =============================================================================
# Helper function: Extract features using CLIP
# =============================================================================
def extract_clip_features_and_labels(image_dir, processor, model):
    """
    Extract feature maps and text labels using the CLIP ViT-Large/14 model.

    Parameters
    ----------
    image_dir : str
        Directory containing subdirectories of images.
    processor : CLIPProcessor
        CLIP processor for preprocessing.
    model : CLIPModel
        CLIP model for feature extraction.

    Returns
    -------
    np.ndarray
        Extracted image feature maps.
    np.ndarray
        Extracted text feature maps (labels).
    """
    image_features = []
    label_features = []

    # Process each subdirectory (class)
    subdirs = sorted(os.listdir(image_dir))
    for subdir in subdirs:
        subdir_path = os.path.join(image_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        # Remove the ID prefix from the subdirectory name to get the label
        label_name = "_".join(subdir.split("_")[1:]).replace("_", " ")
        # print(label_name)

        # Compute text embedding for the class label
        inputs_text = processor(text=label_name, return_tensors="pt", padding=True).to(device)
        try:
            with torch.no_grad():
                outputs = model.get_text_features(**inputs_text)
                text_feature = outputs.pooler_output.detach().cpu().numpy()
                # pooler_output
                # last_hidden_state
        except Exception as e:
            print(f"\n[ERROR] {label_name}")
            print(traceback.format_exc())

        # Process all images in the subdirectory
        image_files = sorted(os.listdir(subdir_path))
        for img_file in image_files:
            img_path = os.path.join(subdir_path, img_file)
            try:
                with Image.open(img_path).convert('RGB') as image:
                    inputs_image = processor(images=image, return_tensors="pt", padding=True).to(device)
                    with torch.no_grad():
                        outputs = model.get_image_features(**inputs_image)
                        image_feature = outputs.pooler_output.detach().cpu().numpy()
                    image_features.append(image_feature)
                    label_features.append(text_feature)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                print(f"\n[ERROR] {img_path}")
                print(traceback.format_exc())

    return np.vstack(image_features), np.vstack(label_features)


if __name__ == '__main__':
    # =============================================================================
    # Extract and save features
    # =============================================================================
    # Train Set
    train_image_dir = os.path.join(base_path, 'datasets/THINGS-EEG/Image/origin/training_images')
    train_image_features, train_label_features = extract_clip_features_and_labels(train_image_dir, processor, model)
    print('Train Image features shape:', train_image_features.shape)
    print('Train Label features shape:', train_label_features.shape)
    save_dir = os.path.join(args.save_path, args.project_dir)
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'clip_feature_maps_image_train.npy'), train_image_features)
    np.save(os.path.join(save_dir, 'clip_feature_maps_label_train.npy'), train_label_features)
    print(f"Train feature extraction complete. Data saved to {save_dir}")

    # Test Set
    test_image_dir = os.path.join(base_path, 'datasets/THINGS-EEG/Image/origin/test_images')
    test_image_features, test_label_features = extract_clip_features_and_labels(test_image_dir, processor, model)
    print('Test Image features shape:', test_image_features.shape)
    print('Test Label features shape:', test_label_features.shape)
    np.save(os.path.join(save_dir, 'clip_feature_maps_image_test.npy'), test_image_features)
    np.save(os.path.join(save_dir, 'clip_feature_maps_label_test.npy'), test_label_features)

    print(f"Test feature extraction complete. Data saved to {save_dir}")

    """
    mask
    """
    # # Train Set
    # train_image_dir = os.path.join(base_path, 'Data/Things_EEG2/Image/mask/training_images')
    # train_image_features, label_features = extract_clip_features_and_labels(train_image_dir, processor, model)
    # print('Image features shape:', train_image_features.shape)
    # test_image_dir = os.path.join(base_path, 'Data/Things_EEG2/Image/mask/test_images')
    # test_image_features, test_label_features = extract_clip_features_and_labels(test_image_dir, processor, model)
    # print('Image features shape:', test_image_features.shape)
    #
    # save_dir = os.path.join(args.save_path, args.project_dir)
    # os.makedirs(save_dir, exist_ok=True)
    # np.save(os.path.join(save_dir, 'clip_feature_maps_mask_image_train.npy'), train_image_features)
    # np.save(os.path.join(save_dir, 'clip_feature_maps_mask_image_test.npy'), test_image_features)
    #
    # print(f"Mask feature extraction complete. Data saved to {save_dir}")
    #
    # train_image_dir = os.path.join(base_path, 'Data/Things_EEG2/Image/mask01/training_images')
    # train_image_features, label_features = extract_clip_features_and_labels(train_image_dir, processor, model)
    # print('Image features shape:', train_image_features.shape)
    # test_image_dir = os.path.join(base_path, 'Data/Things_EEG2/Image/mask01/test_images')
    # test_image_features, test_label_features = extract_clip_features_and_labels(test_image_dir, processor, model)
    # print('Image features shape:', test_image_features.shape)
    #
    # save_dir = os.path.join(args.save_path, args.project_dir)
    # os.makedirs(save_dir, exist_ok=True)
    # np.save(os.path.join(save_dir, 'clip_feature_maps_mask01_image_train.npy'), train_image_features)
    # np.save(os.path.join(save_dir, 'clip_feature_maps_mask01_image_test.npy'), test_image_features)
    #
    # print(f"Mask01 feature extraction complete. Data saved to {save_dir}")


