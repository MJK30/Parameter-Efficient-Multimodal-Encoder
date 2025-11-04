# Parameter-Efficient-Multimodal-Encoder


Parameter-Efficient-Multimodal-Encoder is a PyTorch project for building and training a modern, cross-attention-based model for image-text embedding. The primary design goal is parameter-efficiency, allowing the entire training and experimentation pipeline to run on consumer-grade GPUs (approx. 6GB VRAM).

This project demonstrates how to connect powerful, pre-trained, and frozen unimodal encoders (for vision and text) using a small, trainable "bridge" module.

## Core Idea & Inspiration

This project is inspired by modern adapter-based multimodal architectures, which differ from the classic end-to-end training of models like CLIP.

Instead of training a giant model from scratch, this approach leverages the knowledge of pre-trained "expert" models. The core idea is based on models like BLIP-2, which use a "Q-Former" (a small transformer) to bridge the gap between a frozen vision encoder and a frozen language model.

Our model implements this "Encoder-Bridge" philosophy with a focus on efficiency, using Parameter-Efficient Fine-Tuning (PEFT) techniques.

## Model Architecture

The model uses a two-tower design where image and text data are processed in parallel. The final embeddings are compared outside the model by the loss function. The key to this architecture is that the large encoders are frozen, and only the small adapter/bridge layers are trained.


### Image Tower (Tower 1)
- A[Image: [N, 3, 224, 224]] -> BB(VisionEncoder<br><b>Frozen DINO ViT)
- B --> C[Patch Features [N, 197, 384]];
- C --> D[QueryingBridge];
- D --> E[Image Embedding[N, 512]];  

### Text Tower (Tower 2)
- F[Raw String] -> G(TextEncoder Frozen DistilBERT);
- G --> H[CLS Token<br>[N, 768]];
- H --> I[Projection Head - TRAINABLE];
- I --> J[Text Embedding<br>[N, 512]];


- VisionEncoder (Frozen): A pre-trained DINO ViT-Small model. It takes an image and outputs 197 patch feature vectors (384-dim).

- TextEncoder (Frozen): A pre-trained DistilBERT model. It takes raw text and outputs a single [CLS] token summary vector (768-dim).

- QueryingBridge (Trainable): This is the main trainable module. It uses cross-attention to "query" the 197 patch vectors and fuse them into a single 512-dim summary vector.

- TextEncoder.projection_head (Trainable): A small, trainable MLP that maps the 768-dim text summary to the final 512-dim embedding space.

The training process only updates the QueryingBridge and the TextEncoder.projection_head, leaving the 87M+ parameters of the expert encoders frozen.

## Key Training Techniques

This project is designed to run on 6GB of VRAM by using several key efficiency techniques:

- Parameter-Efficient Fine-Tuning (PEFT): Only a small fraction of the total parameters (the "bridge" and "projectors") are trained.

- Automatic Mixed Precision (AMP): torch.cuda.amp.autocast and GradScaler are used to cut memory usage by using float16.

- Gradient Accumulation: Simulates a large, stable effective batch size of 64 from 8 small "physical" batches of 8.

- Targeted Optimizer: The AdamW optimizer is only given the trainable parameters, saving VRAM by not storing states for the frozen encoders.

## Project Structure

.
├── checkpoints/         # Saved model weights will appear here
├── data/                 # Your Flickr8k data should go here
│   ├── flickr8k_images/
│   └── results.csv
├── config.py             # Stores all hyperparameters and model paths
├── model.py              # Defines the VisionEncoder, TextEncoder, and QueryingBridge
├── dataset.py            # PyTorch Dataset and DataLoader for Flickr30k
├── transforms.py         # Image preprocessing transforms for DINO
├── train.py              # The main training script (Phase 3)




DATA SOURCE - Flickr8k Images https://www.kaggle.com/datasets/adityajn105/flickr8k

