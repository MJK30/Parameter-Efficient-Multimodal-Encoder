import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

import config
from transforms import get_vision_transforms


class Flickr8kDataser(Dataset):
    """
    Pytorch Dataset class for Flickr8k dataset.
    """ 
    
    def __init__(self, split="train"):
        super().__init__()
        
        self.captions_df = pd.read_csv(config.CAPTIONS_PATH)
        
        # Remove rows with missing captions
        self.captions_df = self.captions_df.dropna().reset_index(drop=True)
        
        unique_images = self.captions_df['image'].unique()
        val_images = set(unique_images[-1000:])
        train_images = set(unique_images[:-1000])
        
        if split == "train":
            # Keep rows where the image_name is in the train set
            self.captions_df = self.captions_df[
                self.captions_df['image'].isin(train_images)
            ].reset_index(drop=True)
        elif split == "val":
            # Keep rows where the image_name is in the val set
            self.captions_df = self.captions_df[
                self.captions_df['image'].isin(val_images)
            ].reset_index(drop=True)
        else:
            raise ValueError("Invalid split name. Choose 'train' or 'val'.")
        
        self.image_path_prefix = config.IMAGE_PATH
        
        self.transforms = get_vision_transforms()
        
    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self, index):
        """
        Get the image, caption for a given index

        Args:
            index (_type_): _description_
        """
        
        record = self.captions_df.iloc[index]
        image_name = record['image']
        caption = record['caption']
        
        full_image_path = os.path.join(self.image_path_prefix, image_name)        
        image = Image.open(full_image_path).convert("RGB")        
        image_tensor = self.transforms(image)

        return image_tensor, caption
    
def collate_fn(batch):
    """
    Custom collate function to batch the data samples.

    Args:
        batch (_type_): _description_
    """
    
    images_tensors = [item[0] for item in batch]
    stacked_images = torch.stack(images_tensors, dim=0)
    
    captions = [item[1] for item in batch]

    return stacked_images, captions


def get_dataloader(batch_size=config.PHYSICAL_BATCH_SIZE, num_workers=config.NUM_WORKERS):
    
    
    dataset = Flickr8kDataser()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True, # Pin memory for faster transfers to GPU
        drop_last=True, # Drop the last incomplete batch
    )
    
    return dataloader