import torch
import torch.nn as nn
import timm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import config


"""
Here threee distinct classes are defined -> VisionEncoder, TextEncoder, QueryingBridge 
VisionEncoder - DINO (Small ViT)  is used to generate the patch features. Patch features represent the rich analysis of the image.
QueryBridge - Here the Patch features of the DINO model are processed into the final Image Embedding.
            - A trainable Transformer Decoder layer with cross attention is added to understand the image and generate the Final Image Embedding
TextEncoder - DistilBERT is used to generate the CLS token for the raw string. A trainable Linear Layer is added to downscale to the shared space embedding dimension
"""

class VisionEncoder(nn.Module):
    """
    The images are analysed using the DINO model.
    We use the frozen model to take a preprocessed image.
    Here the output is the patch features of the image(This is not the final image embedding).

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, model_name=config.VISION_MODEL_NAME):
        super().__init__()
        
        # Load the DINO model
        # we dont need to classify anything here
        self.model = timm.create_model(model_name=model_name, pretrained=True, num_classes=0,) 
        
        # Freeze the model parameters for analysis
        # model is in evaluation mode, so trainable parameters are not updated
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, image_tensor):
        """
        Here the Images are analysed
        A preprocessed Image is taken and the patch features are returned as output.
        Input: [N, 3, 224, 244]
        Ouput: [N ,197, 384] -> [Bathch size, Number of patches, Feature dimension]
        
        197 = 1 CLS token + (224/16 * 224/16 = 196 patches)

        Args:
            image_tensor (_type_): _description_
        """
        
        patch_features = self.model.forward_features(image_tensor)
        return patch_features
        
        
class TextEncoder(nn.Module):
    """
    The text is analysed using DistilBERT model.
    Input is a raw text string.
    THe text is preprocessed and analyzed to output token features.

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, model_name=config.TEXT_MODEL_NAME,
                in_dim=config.TEXT_EMBED_DIM,
                out_dim=config.FINAL_EMBED_DIM):
        super().__init__()
        
        # Tokenizer to preprocess the text
        # Load DistilBERT model and it´s vocabulary
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # anylsis step
        # loads the frozen DistilBERT model
        self.model = AutoModel.from_pretrained(model_name)
        
        # freeze all the parameters of the model
        # run in evaluation mode for analysis
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        # projection layer to project the text features to the final embedding dimension
        # small MLP to map to 768 dim BERT
        # features to our final embedding dimension
        self.projection_head = nn.Sequential(
            nn.Linear(in_dim, in_dim*2), # Linear(768, 1536) 
            nn.GELU(), # -> GELU )
            nn.Linear(in_dim*2, out_dim) # -> Linear(1536, 512
        )
        
    def forward(self, text_list: list[str]):
        """
        Input: a list of raw strings
        Output: Final embedding        

        Args:
            text_list (list[str]): _description_
        """
        
        # Preprocessing the text input
        # adds paddings
        # truncates if too long
        # returns PyTorch tensors
        tokenized_inputs = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(config.DEVICE)
        
        # Analyse the text
        # Pass the text through DistilBERT model
        # [N, L] -> [N, L, 768] ([Batch size, Sequence length, Feature dimension])
        text_outputs = self.model(**tokenized_inputs)
        
        # Get CLS token feature - token describing the whole sequence
        # token at index 0
        # [N, L , 768] -> [N, 768]
        text_summary_features = text_outputs.last_hidden_state[:, 0, :]
        
        # Project the text features to the final embedding dimension
        # [N, 768] -> [N, 512]
        final_text_embeddings = self.projection_head(text_summary_features)
        return final_text_embeddings
    
    
class QueryingBridge(nn.Module):
    """
    DINO Model analyses the image and outputs patch features.
    DINO patch features are projected to the bridge query, where self attention, cross attention and MLP layers are applied.
    The output is the final image embedding.
    The querying bridge uses cross attention to relate the image patch features to learnable query tokens.
    This larger context is then summarized to obtain the final image embedding.
    

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self,
                in_dim=config.VISION_EMBED_DIM, # 384
                bridge_dim=config.BRIDGE_EMBED_DIM, # 768
                out_dim=config.FINAL_EMBED_DIM, # 512
                num_queries=config.NUM_QUERIES, 
                num_layers=config.NUM_BRIDGE_LAYERS,):
        super().__init__()
        
        # Preprocessing layer to map DINO features to bridge dimension
        # This projects the 384 dim DINO features to 768 dim BERT features
        self.vision_input_projection = nn.Linear(in_dim, bridge_dim)
        
        
        # Query tokens to study the image features
        # these are learnable parameters
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, bridge_dim))
        
        # Here DINO Patch features are converted to Final Image Embedding
        # This layer has a self attention mechanism to refine the queries
        # and cross attention mechanism to attend to the image patch features
        # MLP layer to understand and summarize the features
        # Multiple layers are stacked and the DINO patch features are related to the queries to obtain FINALL IMAGE EMBEDDING
        self.bridge_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=bridge_dim,
                nhead=8,
                dim_feedforward=bridge_dim*4,
                batch_first=True,
            ) for _ in range(num_layers)
        ])
        
        # This Linear Layer projects the Image Embedding to the Final shared dimension with Text encoding
        # maps the final 768 dim features to 512 dim features
        self.projection_head = nn.Sequential(
            nn.Linear(bridge_dim, bridge_dim*2),  # Linear(768, 1536)
            nn.GELU(),                          # -> GELU
            nn.Linear(bridge_dim*2, out_dim)   # -> Linear(1536, 512)
        )
        
        # Learnable logit scaling parameter
        # used in contrastive learning loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_patch_features):
        """
        takes the patch features from the DINO model
        Input; [N, 197, 384]
        Ouput; [N, 512] -> This is the final image embedding

        Args:
            image_patch_features (_type_): _description_
        """
        
        # Preprocess the DINO features to bridge dimension
        # [N, 197, 384] -> [N, 197, 768]
        projected_patches = self.vision_input_projection(image_patch_features)
        
        # Expand the query tokens to match the batch size
        # [1, num_queries, 768] -> [N, num_queries, 768]
        queries = self.query_tokens.expand(projected_patches.shape[0], -1, -1)
        
        # num_queries attend to the 197 image patches
        # running for the num,_layers defined
        for layer in self.bridge_layers:
            queries = layer(tgt=queries, memory=projected_patches)
            
        
        # average the queries
        # a single image embedding is obtained
        image_summary = queries.mean(dim=1)  # [N, num_queries, 768] -> [N, 768]
        
        # Project to final embedding dimension
        final_image_embeddings = self.projection_head(image_summary)  # [N, 768] -> [N, 512]
        return final_image_embeddings
        

