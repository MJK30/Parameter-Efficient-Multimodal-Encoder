import torch
from torchvision import transforms
import config


# Defining the Normalization constants for DINO pretrained ViT model
# DINO was trained on the following mean and std values, so we use them here for normalization
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]


def get_vision_transforms():
    """
    Transform any image format given as the input to the standard format for the DINO model [3,224,224]
    Image is resized to 224x224 input, and then converted to Tensors
    Tensors are normalized using DINO stats
    """
    
    vision_transforms = transforms.Compose([
        # Resize the shortest side to 224, maintaining aspect ratio
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        # Center Crop to 224x224
        transforms.CenterCrop(224),
        # Convert image to Tensor
        transforms.ToTensor(),
        # Normalize using DINO mean and std
        transforms.Normalize(mean=DINO_MEAN, std=DINO_STD)
    ])
    return vision_transforms