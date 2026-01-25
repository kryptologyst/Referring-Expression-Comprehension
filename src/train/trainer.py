"""
Training module for referring expression comprehension models.

This module provides training loops, loss functions, and optimization
utilities for referring expression comprehension tasks.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.core import AverageMeter, format_time, get_lr
from ..eval.metrics import ReferringExpressionEvaluator


class ReferringExpressionLoss(nn.Module):
    """Loss function for referring expression comprehension."""
    
    def __init__(
        self,
        bbox_weight: float = 1.0,
        confidence_weight: float = 1.0,
        iou_weight: float = 0.5,
    ):
        """Initialize loss function.
        
        Args:
            bbox_weight: Weight for bounding box regression loss
            confidence_weight: Weight for confidence prediction loss
            iou_weight: Weight for IoU loss
        """
        super().__init__()
        self.bbox_weight = bbox_weight
        self.confidence_weight = confidence_weight
        self.iou_weight = iou_weight
        
        self.bbox_loss = nn.SmoothL1Loss()
        self.confidence_loss = nn.BCELoss()
    
    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_confidences: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_confidences: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss.
        
        Args:
            pred_bboxes: Predicted bounding boxes [B, N, 4]
            pred_confidences: Predicted confidence scores [B, N]
            target_bboxes: Target bounding boxes [B, 4]
            target_confidences: Target confidence scores [B] (optional)
            
        Returns:
            Dictionary containing loss components
        """
        batch_size = pred_bboxes.shape[0]
        device = pred_bboxes.device
        
        # Find best predictions for each sample
        best_indices = torch.argmax(pred_confidences, dim=1)  # [B]
        best_bboxes = pred_bboxes[torch.arange(batch_size), best_indices]  # [B, 4]
        best_confidences = pred_confidences[torch.arange(batch_size), best_indices]  # [B]
        
        # Bounding box regression loss
        bbox_loss = self.bbox_loss(best_bboxes, target_bboxes)
        
        # Confidence loss
        if target_confidences is None:
            # Use IoU as target confidence
            target_confidences = self._calculate_iou_targets(best_bboxes, target_bboxes)
        
        confidence_loss = self.confidence_loss(best_confidences, target_confidences)
        
        # IoU loss (additional supervision)
        iou_loss = self._calculate_iou_loss(best_bboxes, target_bboxes)
        
        # Total loss
        total_loss = (
            self.bbox_weight * bbox_loss +
            self.confidence_weight * confidence_loss +
            self.iou_weight * iou_loss
        )
        
        return {
            "total_loss": total_loss,
            "bbox_loss": bbox_loss,
            "confidence_loss": confidence_loss,
            "iou_loss": iou_loss,
        }
    
    def _calculate_iou_targets(
        self,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate IoU-based target confidences.
        
        Args:
            pred_bboxes: Predicted bounding boxes [B, 4]
            target_bboxes: Target bounding boxes [B, 4]
            
        Returns:
            IoU-based target confidences [B]
        """
        batch_size = pred_bboxes.shape[0]
        ious = torch.zeros(batch_size, device=pred_bboxes.device)
        
        for i in range(batch_size):
            ious[i] = self._calculate_iou(pred_bboxes[i], target_bboxes[i])
        
        return ious
    
    def _calculate_iou_loss(
        self,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate IoU loss.
        
        Args:
            pred_bboxes: Predicted bounding boxes [B, 4]
            target_bboxes: Target bounding boxes [B, 4]
            
        Returns:
            IoU loss
        """
        batch_size = pred_bboxes.shape[0]
        ious = torch.zeros(batch_size, device=pred_bboxes.device)
        
        for i in range(batch_size):
            ious[i] = self._calculate_iou(pred_bboxes[i], target_bboxes[i])
        
        # IoU loss: maximize IoU (minimize 1 - IoU)
        return torch.mean(1.0 - ious)
    
    def _calculate_iou(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two bounding boxes."""
        # Convert to [x1, y1, x2, y2] format
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = torch.max(x1_1, x1_2)
        y1_i = torch.max(y1_1, y1_2)
        x2_i = torch.min(x2_1, x2_2)
        y2_i = torch.min(y2_1, y2_2)
        
        intersection = torch.clamp(x2_i - x1_i, min=0) * torch.clamp(y2_i - y1_i, min=0)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / torch.clamp(union, min=1e-6)


class ReferringExpressionTrainer:
    """Trainer for referring expression comprehension models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: torch.device,
        config: Dict[str, Any],
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            device: Device to train on
            config: Training configuration
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.config = config
        
        # Setup loss function
        self.criterion = ReferringExpressionLoss(
            bbox_weight=config.get("bbox_weight", 1.0),
            confidence_weight=config.get("confidence_weight", 1.0),
            iou_weight=config.get("iou_weight", 0.5),
        )
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4),
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("num_epochs", 100),
        )
        
        # Setup evaluator
        self.evaluator = ReferringExpressionEvaluator(
            iou_threshold=config.get("iou_threshold", 0.5)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_metrics = []
        
        # Create output directory
        self.output_dir = config.get("output_dir", "outputs")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        
        # Initialize meters
        loss_meter = AverageMeter()
        bbox_loss_meter = AverageMeter()
        confidence_loss_meter = AverageMeter()
        iou_loss_meter = AverageMeter()
        
        # Training loop
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch["images"].to(self.device)
            texts = batch["texts"]
            target_bboxes = batch["bboxes"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, texts)
            
            # Compute loss
            loss_dict = self.criterion(
                outputs["bboxes"],
                outputs["confidences"],
                target_bboxes,
            )
            
            # Backward pass
            loss_dict["total_loss"].backward()
            
            # Gradient clipping
            if self.config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["grad_clip"]
                )
            
            self.optimizer.step()
            
            # Update meters
            batch_size = images.shape[0]
            loss_meter.update(loss_dict["total_loss"].item(), batch_size)
            bbox_loss_meter.update(loss_dict["bbox_loss"].item(), batch_size)
            confidence_loss_meter.update(loss_dict["confidence_loss"].item(), batch_size)
            iou_loss_meter.update(loss_dict["iou_loss"].item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_meter.avg:.4f}",
                "bbox": f"{bbox_loss_meter.avg:.4f}",
                "conf": f"{confidence_loss_meter.avg:.4f}",
                "iou": f"{iou_loss_meter.avg:.4f}",
                "lr": f"{get_lr(self.optimizer):.2e}",
            })
        
        return {
            "train_loss": loss_meter.avg,
            "train_bbox_loss": bbox_loss_meter.avg,
            "train_confidence_loss": confidence_loss_meter.avg,
            "train_iou_loss": iou_loss_meter.avg,
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate the model.
        
        Returns:
            Dictionary containing validation metrics
        """
        return self.evaluator.evaluate(
            self.model,
            self.val_dataloader,
            self.device,
        )
    
    def train(self) -> None:
        """Train the model for the specified number of epochs."""
        num_epochs = self.config.get("num_epochs", 100)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics)
            
            # Validate
            val_metrics = self.validate()
            self.val_metrics.append(val_metrics)
            
            # Update scheduler
            self.scheduler.step()
            
            # Print epoch results
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val IoU: {val_metrics['mean_iou']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            print(f"Learning Rate: {get_lr(self.optimizer):.2e}")
            
            # Save best model
            if val_metrics["accuracy"] > self.best_accuracy:
                self.best_accuracy = val_metrics["accuracy"]
                self.save_checkpoint(is_best=True)
                print(f"New best accuracy: {self.best_accuracy:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get("save_every", 10) == 0:
                self.save_checkpoint(is_best=False)
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(total_time)}")
        print(f"Best validation accuracy: {self.best_accuracy:.4f}")
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_accuracy": self.best_accuracy,
            "config": self.config,
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.output_dir, "checkpoint.pth")
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_accuracy = checkpoint["best_accuracy"]
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Best accuracy: {self.best_accuracy:.4f}")
