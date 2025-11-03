import torch

"""
Central File to define the configurations of the models used.
data locations are defined here.

For Image Encoding -> Small Vision Transformer is used -> DINO(v1)
DINO(v1) has 21M params

For Text Encoding -> Small BERT Transformer is used -> DistilBERT
DistilBERT has 66M param

"""
    
    
# Models Definition
VISION_MODEL_NAME = "vit_small_patch16_224.dino" #21M params
TEXT_MODEL_NAME = "distilbert-base-uncased" #66M params

# Dimensions for data
VISION_EMBED_DIM = 384 # ViT Output Dimension
TEXT_EMBED_DIM = 768 # DistilBERT Output Dimension
BRIDGE_EMBED_DIM = 768 # Common Dimension for Cross Attention (Based on Text Embedding Dimension)
FINAL_EMBED_DIM = 512 # Final Shared Embedding Space

# Cross Attention Configurations
NUM_QUERIES = 32 # 32 Queries Tokens
NUM_BRIDGE_LAYERS = 4 # 4 Layers of Cross Attention

# Training Configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PHYSICAL_BATCH_SIZE = 8 # (image,text) pairs for training
ACCUMULATION_STEPS = 8 # Effective Batch Size after Gradient Accumulation
LEARNING_RATE = 1e-4
EPOCHS = 20
NUM_WORKERS = 4 # For Data Loading

# Data Paths
DATA_PATH = "data/"
IMAGE_PATH = "data/Images"
CAPTIONS_PATH = "data/results.csv"




