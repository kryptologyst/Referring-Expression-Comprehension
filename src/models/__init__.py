"""
Advanced models for referring expression comprehension.

This module implements state-of-the-art models including CLIP-based approaches,
MDETR, LAVT, and RefTR for referring expression comprehension tasks.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer, AutoModel


class CLIPReferringExpressionModel(nn.Module):
    """CLIP-based referring expression comprehension model.
    
    This model uses CLIP's vision and text encoders to perform referring
    expression comprehension by computing similarity between image regions
    and referring expressions.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        num_queries: int = 100,
        hidden_dim: int = 256,
    ):
        """Initialize CLIP-based referring expression model.
        
        Args:
            model_name: CLIP model name
            num_queries: Number of object queries
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Freeze CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Projection layers
        self.image_proj = nn.Linear(self.clip_model.config.vision_config.hidden_size, hidden_dim)
        self.text_proj = nn.Linear(self.clip_model.config.text_config.hidden_size, hidden_dim)
        
        # Object queries
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Cross-attention for referring expression comprehension
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Output heads
        self.bbox_head = nn.Linear(hidden_dim, 4)  # [x, y, w, h]
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
    def forward(
        self,
        images: torch.Tensor,
        texts: List[str],
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for referring expression comprehension.
        
        Args:
            images: Input images [B, C, H, W]
            texts: List of referring expressions
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing predictions and optional attention weights
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Process images and texts with CLIP
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(device)
        
        # Get CLIP features
        clip_outputs = self.clip_model(**inputs)
        image_features = clip_outputs.image_embeds  # [B, hidden_size]
        text_features = clip_outputs.text_embeds    # [B, hidden_size]
        
        # Project features
        image_proj = self.image_proj(image_features)  # [B, hidden_dim]
        text_proj = self.text_proj(text_features)     # [B, hidden_dim]
        
        # Object queries
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention between queries and text
        attn_output, attn_weights = self.cross_attn(
            queries, text_proj.unsqueeze(1), text_proj.unsqueeze(1)
        )
        
        # Generate predictions
        bbox_pred = self.bbox_head(attn_output)  # [B, num_queries, 4]
        confidence_pred = self.confidence_head(attn_output).squeeze(-1)  # [B, num_queries]
        
        # Apply sigmoid to confidence
        confidence_pred = torch.sigmoid(confidence_pred)
        
        results = {
            "bboxes": bbox_pred,
            "confidences": confidence_pred,
            "image_features": image_proj,
            "text_features": text_proj,
        }
        
        if return_attention:
            results["attention_weights"] = attn_weights
            
        return results


class MDETRReferringExpressionModel(nn.Module):
    """MDETR-based referring expression comprehension model.
    
    This model implements a modified version of MDETR (Multimodal DEtection TRansformer)
    specifically adapted for referring expression comprehension tasks.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
    ):
        """Initialize MDETR-based model.
        
        Args:
            hidden_dim: Hidden dimension size
            num_queries: Number of object queries
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Vision encoder (simplified ResNet backbone)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, 3, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Text encoder (simplified transformer)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output heads
        self.bbox_head = nn.Linear(hidden_dim, 4)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for MDETR model.
        
        Args:
            images: Input images [B, C, H, W]
            text_tokens: Text token embeddings [B, L, hidden_dim]
            text_mask: Text attention mask [B, L]
            
        Returns:
            Dictionary containing predictions
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Vision encoding
        vision_features = self.vision_encoder(images)  # [B, hidden_dim, H', W']
        B, C, H, W = vision_features.shape
        
        # Flatten spatial dimensions
        vision_features = vision_features.flatten(2).transpose(1, 2)  # [B, H'*W', hidden_dim]
        
        # Add positional encoding
        pos_encoding = self.pos_encoding[:, :H*W, :].expand(batch_size, -1, -1)
        vision_features = vision_features + pos_encoding
        
        # Text encoding
        text_features = self.text_encoder(text_tokens, src_key_padding_mask=text_mask)
        
        # Object queries
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Transformer decoder
        decoder_output = self.transformer_decoder(
            queries, vision_features
        )
        
        # Generate predictions
        bbox_pred = self.bbox_head(decoder_output)  # [B, num_queries, 4]
        confidence_pred = self.confidence_head(decoder_output).squeeze(-1)  # [B, num_queries]
        
        # Apply sigmoid to confidence
        confidence_pred = torch.sigmoid(confidence_pred)
        
        return {
            "bboxes": bbox_pred,
            "confidences": confidence_pred,
            "vision_features": vision_features,
            "text_features": text_features,
        }


class LAVTReferringExpressionModel(nn.Module):
    """LAVT (Language-Aware Vision Transformer) for referring expression comprehension.
    
    This model implements a vision transformer that incorporates language
    information at multiple levels for better referring expression understanding.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_queries: int = 100,
        num_heads: int = 12,
        num_layers: int = 12,
        patch_size: int = 16,
        dropout: float = 0.1,
    ):
        """Initialize LAVT model.
        
        Args:
            hidden_dim: Hidden dimension size
            num_queries: Number of object queries
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            patch_size: Patch size for vision transformer
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Language-aware transformer layers
        self.transformer_layers = nn.ModuleList([
            LanguageAwareTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Output heads
        self.bbox_head = nn.Linear(hidden_dim, 4)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
    def forward(
        self,
        images: torch.Tensor,
        text_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for LAVT model.
        
        Args:
            images: Input images [B, C, H, W]
            text_features: Text features [B, L, hidden_dim]
            
        Returns:
            Dictionary containing predictions
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Patch embedding
        x = self.patch_embed(images)  # [B, hidden_dim, H', W']
        B, C, H, W = x.shape
        
        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', hidden_dim]
        
        # Add positional encoding
        pos_encoding = self.pos_embed[:, :H*W, :].expand(batch_size, -1, -1)
        x = x + pos_encoding
        
        # Apply language-aware transformer layers
        for layer in self.transformer_layers:
            x = layer(x, text_features)
        
        # Object queries
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention between queries and image features
        attn_output, _ = nn.MultiheadAttention(
            self.hidden_dim, 8, batch_first=True
        )(queries, x, x)
        
        # Generate predictions
        bbox_pred = self.bbox_head(attn_output)  # [B, num_queries, 4]
        confidence_pred = self.confidence_head(attn_output).squeeze(-1)  # [B, num_queries]
        
        # Apply sigmoid to confidence
        confidence_pred = torch.sigmoid(confidence_pred)
        
        return {
            "bboxes": bbox_pred,
            "confidences": confidence_pred,
            "image_features": x,
            "text_features": text_features,
        }


class LanguageAwareTransformerLayer(nn.Module):
    """Language-aware transformer layer for LAVT model."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        """Initialize language-aware transformer layer.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for language-aware transformer layer.
        
        Args:
            x: Image features [B, N, hidden_dim]
            text_features: Text features [B, L, hidden_dim]
            
        Returns:
            Updated image features
        """
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Cross-attention with text
        cross_output, _ = self.cross_attn(x, text_features, text_features)
        x = self.norm2(x + cross_output)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = self.norm3(x + ffn_output)
        
        return x


class RefTRReferringExpressionModel(nn.Module):
    """RefTR (Referring Expression Transformer) model.
    
    This model implements a transformer-based architecture specifically
    designed for referring expression comprehension with attention mechanisms
    that focus on relevant image regions based on text descriptions.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_queries: int = 100,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        """Initialize RefTR model.
        
        Args:
            hidden_dim: Hidden dimension size
            num_queries: Number of object queries
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Feature extraction
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, 3, 2, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Object queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        # Referring expression attention
        self.ref_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Output heads
        self.bbox_head = nn.Linear(hidden_dim, 4)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for RefTR model.
        
        Args:
            images: Input images [B, C, H, W]
            text_tokens: Text token embeddings [B, L, hidden_dim]
            text_mask: Text attention mask [B, L]
            
        Returns:
            Dictionary containing predictions
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Image encoding
        image_features = self.image_encoder(images)  # [B, hidden_dim, H', W']
        B, C, H, W = image_features.shape
        
        # Flatten spatial dimensions
        image_features = image_features.flatten(2).transpose(1, 2)  # [B, H'*W', hidden_dim]
        
        # Add positional encoding
        pos_encoding = self.pos_encoding[:, :H*W, :].expand(batch_size, -1, -1)
        image_features = image_features + pos_encoding
        
        # Text encoding
        text_features = self.text_encoder(text_tokens, src_key_padding_mask=text_mask)
        
        # Object queries
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Referring expression attention
        attn_output, attn_weights = self.ref_attn(
            queries, text_features, text_features, key_padding_mask=text_mask
        )
        
        # Generate predictions
        bbox_pred = self.bbox_head(attn_output)  # [B, num_queries, 4]
        confidence_pred = self.confidence_head(attn_output).squeeze(-1)  # [B, num_queries]
        
        # Apply sigmoid to confidence
        confidence_pred = torch.sigmoid(confidence_pred)
        
        return {
            "bboxes": bbox_pred,
            "confidences": confidence_pred,
            "image_features": image_features,
            "text_features": text_features,
            "attention_weights": attn_weights,
        }
