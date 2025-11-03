# Parameter-Efficient-Multimodal-Encoder


This is a PyTorch project to build and train a light cross-attention fusion model for image-text embedding. It is designed from the ground up to be parameter-efficient and runnable on consumer-grade GPUs (6GB VRAM).

Inspiration

This project began as an exploration of OpenAI's CLIP model, the foundational two-tower model for aligning images and text. However, CLIP's architecture, while powerful, trains a giant vision encoder from scratch, which is computationally massive.

The inspiration for this project came from more recent "Encoder-Bridge" or "Adapter" models, which ask a new question:

"Instead of training a giant model from scratch, can we take existing, pre-trained 'expert' models (for vision and text) and just train a tiny, lightweight 'bridge' to connect them?"

This is the core idea behind state-of-the-art models like BLIP-2, which uses a "Q-Former" to connect a frozen vision encoder to a frozen Large Language Model.

Model Architecture

Our model is a "Two-Tower" design where the comparison only happens in the final loss function. However, the Image Tower is a sophisticated two-stage system.

graph TD
    subgraph Image Tower (Tower 1)
        direction TB
        A[Image: [N, 3, 224, 224]] --> B(VisionEncoder<br><b>Frozen DINO ViT-S</b>);
        B --> C[Patch Features<br>[N, 197, 384]];
        C --> D[QueryingBridge<br><b>TRAINABLE</b>];
        D --> E[Image Embedding<br>[N, 512]];
    end

    subgraph Text Tower (Tower 2)
        direction TB
        F[Text: ["a dog..."]] --> G(TextEncoder<br><b>Frozen DistilBERT</b>);
        G --> H[CLS Token<br>[N, 768]];
        H --> I[Projection Head<br><b>TRAINABLE</b>];
        I --> J[Text Embedding<br>[N, 512]];
    end

    subgraph Loss (Outside Model)
        direction LR
        E --> K{InfoNCE<br>Contrastive Loss};
        J --> K;
    end


Frozen VisionEncoder (DINO): A pre-trained DINO ViT-Small model that acts as our "expert eye." It provides 197 patch-based "visual words" (384-dim).

Frozen TextEncoder (DistilBERT): A pre-trained DistilBERT model that acts as our "expert reader." It provides a single [CLS] token summary (768-dim).

Trainable QueryingBridge: This is the "brain" of our project. This module (which includes cross-attention layers, learnable queries, and a projection head) learns to take DINO's 197 "visual words" and "summarize" them into a single 512-dim vector.

Trainable TextEncoder.projection_head: A tiny MLP that learns to "translate" DistilBERT's 768-dim summary into the same 512-dim shared space.

Training only involves updating the QueryingBridge and the TextEncoder.projection_head, leaving the 87M+ parameters of the expert encoders untouched.

✨ Key Features & Techniques

This project runs on 6GB of VRAM thanks to a combination of modern techniques:

Parameter-Efficient Fine-Tuning (PEFT): Only a small fraction of the total parameters (the "bridge" and "projectors") are trained.

Automatic Mixed Precision (AMP): torch.cuda.amp.autocast and GradScaler are used to cut memory usage nearly in half by using float16.

Gradient Accumulation: We simulate a large, stable effective batch size of 64 by accumulating gradients over 8 small "physical" batches of 8.

Targeted Optimizer: The AdamW optimizer is only given the trainable parameters, saving a massive amount of VRAM by not storing optimizer states for the frozen encoders.

Project Structure

.
├── checkpoints/         # Saved model weights will appear here
├── data/                 # Your Flickr30k data should go here
│   ├── flickr30k_images/
│   └── results.csv
├── config.py             # The main control panel (model names, batch size, LR)
├── model.py              # The 3 core nn.Module classes (VisionEncoder, TextEncoder, QueryingBridge)
├── dataset.py            # The Flickr30k Dataset and DataLoader
├── transforms.py         # Image preprocessing transforms for DINO
├── train.py              # The main training script (Phase 3)
├── eval.py               # The evaluation script (Phase 4)
└── verify_memory.py      # The hardware validation script (Phase 1)

