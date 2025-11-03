import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import datetime

import config
from dataset import get_dataloader
from model import VisionEncoder, TextEncoder, QueryingBridge


def compute_loss(image_embeds, text_embeds, logit_scale):
    """
    # Compute the contrastive loss between image and text embeddings
    # using cosine similarity and symmetric cross-entropy loss.
    
    # 
    
    """
    # Normalize the embeddings
    image_embeds_norms = F.normalize(image_embeds, p=2, dim=-1)
    text_embeds_norms = F.normalize(text_embeds, p=2, dim=-1)
    
    # calculate the cosine similarity -> dot product of the normalized text and image embeddings
    # use logit_scale as a scaling factor
    logits = (image_embeds_norms @ text_embeds_norms.T) * logit_scale.exp()
    
    # Create labels for contrastive learning
    # for a batch size of 8, the labels are [0, 1, 2, ..., 7]
    labels = torch.arange(logits.shape[0]).to(config.DEVICE) # tells the model which image corresponds to which text
    
    # calculate the symmetric cross-entropy loss
    loss_i2t = F.cross_entropy(logits, labels) # image to text
    loss_t2i = F.cross_entropy(logits.T, labels) # text to image
    
    # average the bidirectional losses
    loss = (loss_i2t + loss_t2i) / 2.0
    
    return loss

def train():
    """
    The main training code
    FORWARD PASS:
    1. Images are passed through the frozen vision encoder(DINO) to get patch features
    2. The patch features are passed through the querying bridge to get the final image embeddings
    3. Texts are passed through the frozen text encoder(BERT) to get text embeddings
    4. Contrastive loss is computed between image and text embeddings
    BACKWARD PASS:
    1. Gradients are computed and scaled using GradScaler for mixed precision training
    2. Optimizer updates the weights of the querying bridge and text projection head (only trainable parts, avoiding frozen models)
    3. Checkpoints are saved after each epoch
    
    """
    
    print("Start Training")    
    
    # create directory to save checkpoints
    os.makedirs("checkpoints", exist_ok=True)
    
    # load the models
    vision_encoder = VisionEncoder().to(config.DEVICE)
    text_encoder = TextEncoder().to(config.DEVICE)
    querying_bridge = QueryingBridge().to(config.DEVICE)
    
    # set the models to eval mode
    vision_encoder.eval()
    text_encoder.eval() # freeze the BERT model weights
    
    # 
    text_encoder.projection_head.train()
    querying_bridge.train()
    
    # load the data
    print("Loading Data")
    dataloader = get_dataloader()
    
    # define the optimizer
    # pass only the trainable parameters to the optimizer
    trainable_params = list(text_encoder.projection_head.parameters()) + list(querying_bridge.parameters())
    optimizer = Adam(trainable_params, lr=config.LEARNING_RATE, weight_decay=1e-3)
    
    # gradient scaler for mixed precision training
    scaler = GradScaler()
    
    print("Beginning Training Loop")
    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1} / {config.EPOCHS}")
        
        querying_bridge.train()
        text_encoder.projection_head.train()
        
        total_loss = 0.0
        
        # reset the optimizer at the start of each epoch
        optimizer.zero_grad()
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}",)
        for i, (images, texts) in enumerate(loop):
            
            images = images.to(config.DEVICE)
            
            # FORWARD PASS
            # use autocast for mixed precision (float16) -> speeds up training and reduces memory usage
            # 
            with autocast():
                
                # IMAGE
                with torch.no_grad(): # freeze the vision encoder
                    image_features = vision_encoder(images) # [N, 3, 224, 224] -> [N, 197, 384]
                image_embeds = querying_bridge(image_features) # [N, 197, 384] -> [N, 512]; Trainable part
                
                # TEXT
                text_features = text_encoder(texts) 
                
                # LOSS COMPUTATION
                # compute the contrastive loss between image and text embeddings
                logit_scale = querying_bridge.logit_scale
                loss = compute_loss(image_embeds, text_features, logit_scale)
                
                loss = loss / config.ACCUMULATION_STEPS # normalize the loss for gradient accumulation -> helps with stability
                
                
            
            # BACKWARD PASS
            # Scale the loss and call backward()
            scaler.scale(loss).backward() # calculate the gradients. Prevents the zero gradients problem
    
            # Update the weights -> optimizer step every ACCUMULATION_STEPS
            if (i + 1) % config.ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
            
            total_loss += loss.item() * config.ACCUMULATION_STEPS
            loop.set_postfix(loss=total_loss / ((i + 1) * config.PHYSICAL_BATCH_SIZE))
            
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.6f}")
        
        # Save the model checkpoint
        checkpoint_path = f"checkpoints/epoch_{epoch+1}.pt"
        # only the state_dict of the trainable parts saved
        torch.save({
            'querying_bridge_state_dict': querying_bridge.state_dict(),
            'text_encoder_projection_head_state_dict': text_encoder.projection_head.state_dict(),
        }, checkpoint_path
        )
        
    print("Training Completed")
    
if __name__ == "__main__":
    train()
    
    
