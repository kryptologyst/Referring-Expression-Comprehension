"""
Evaluation metrics and utilities for referring expression comprehension.

This module provides comprehensive evaluation metrics including accuracy,
IoU, precision, recall, and F1 score for referring expression comprehension tasks.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class ReferringExpressionMetrics:
    """Comprehensive evaluation metrics for referring expression comprehension."""
    
    def __init__(self, iou_threshold: float = 0.5):
        """Initialize metrics calculator.
        
        Args:
            iou_threshold: IoU threshold for considering a prediction correct
        """
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.total_samples = 0
        self.correct_predictions = 0
        self.total_iou = 0.0
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
        
    def update(
        self,
        pred_bboxes: torch.Tensor,
        pred_confidences: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_confidences: Optional[torch.Tensor] = None,
    ) -> None:
        """Update metrics with new predictions and targets.
        
        Args:
            pred_bboxes: Predicted bounding boxes [B, N, 4]
            pred_confidences: Predicted confidence scores [B, N]
            target_bboxes: Target bounding boxes [B, 4]
            target_confidences: Target confidence scores [B] (optional)
        """
        batch_size = pred_bboxes.shape[0]
        
        for i in range(batch_size):
            # Get best prediction for this sample
            best_idx = torch.argmax(pred_confidences[i])
            pred_bbox = pred_bboxes[i, best_idx]
            pred_conf = pred_confidences[i, best_idx]
            
            target_bbox = target_bboxes[i]
            
            # Calculate IoU
            iou = self._calculate_iou(pred_bbox, target_bbox)
            self.total_iou += iou
            
            # Check if prediction is correct
            is_correct = iou >= self.iou_threshold
            if is_correct:
                self.correct_predictions += 1
            
            # Calculate precision, recall, F1
            precision, recall, f1 = self._calculate_prf1(
                pred_bbox, pred_conf, target_bbox, iou
            )
            
            self.precision_scores.append(precision)
            self.recall_scores.append(recall)
            self.f1_scores.append(f1)
            
            self.total_samples += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics.
        
        Returns:
            Dictionary containing all computed metrics
        """
        if self.total_samples == 0:
            return {
                "accuracy": 0.0,
                "mean_iou": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }
        
        accuracy = self.correct_predictions / self.total_samples
        mean_iou = self.total_iou / self.total_samples
        precision = np.mean(self.precision_scores)
        recall = np.mean(self.recall_scores)
        f1 = np.mean(self.f1_scores)
        
        return {
            "accuracy": accuracy,
            "mean_iou": mean_iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    def _calculate_iou(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x, y, w, h]
            bbox2: Second bounding box [x, y, w, h]
            
        Returns:
            IoU value
        """
        # Convert to [x1, y1, x2, y2] format
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_prf1(
        self,
        pred_bbox: torch.Tensor,
        pred_conf: torch.Tensor,
        target_bbox: torch.Tensor,
        iou: float,
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score.
        
        Args:
            pred_bbox: Predicted bounding box
            pred_conf: Predicted confidence
            target_bbox: Target bounding box
            iou: IoU between prediction and target
            
        Returns:
            Tuple of (precision, recall, f1)
        """
        # For referring expression comprehension, we consider:
        # - True Positive: IoU >= threshold
        # - False Positive: IoU < threshold but high confidence
        # - False Negative: IoU < threshold and low confidence
        
        tp = 1 if iou >= self.iou_threshold else 0
        fp = 1 if iou < self.iou_threshold and pred_conf > 0.5 else 0
        fn = 1 if iou < self.iou_threshold and pred_conf <= 0.5 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1


class ReferringExpressionEvaluator:
    """High-level evaluator for referring expression comprehension models."""
    
    def __init__(self, iou_threshold: float = 0.5):
        """Initialize evaluator.
        
        Args:
            iou_threshold: IoU threshold for correctness
        """
        self.iou_threshold = iou_threshold
        self.metrics = ReferringExpressionMetrics(iou_threshold)
    
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate model on dataset.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader for evaluation
            device: Device to run evaluation on
            
        Returns:
            Dictionary containing evaluation metrics
        """
        model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch["images"].to(device)
                texts = batch["texts"]
                target_bboxes = batch["bboxes"].to(device)
                
                # Get model predictions
                if hasattr(model, 'forward'):
                    outputs = model(images, texts)
                else:
                    # Handle different model interfaces
                    outputs = model(images, texts)
                
                pred_bboxes = outputs["bboxes"]
                pred_confidences = outputs["confidences"]
                
                # Update metrics
                self.metrics.update(
                    pred_bboxes, pred_confidences, target_bboxes
                )
        
        return self.metrics.compute()
    
    def evaluate_single(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        text: str,
        target_bbox: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate model on a single sample.
        
        Args:
            model: Model to evaluate
            image: Input image [C, H, W]
            text: Referring expression
            target_bbox: Target bounding box [4]
            device: Device to run evaluation on
            
        Returns:
            Dictionary containing evaluation metrics for this sample
        """
        model.eval()
        
        with torch.no_grad():
            # Add batch dimension
            image = image.unsqueeze(0).to(device)
            target_bbox = target_bbox.unsqueeze(0).to(device)
            
            # Get model predictions
            outputs = model(image, [text])
            
            pred_bboxes = outputs["bboxes"]
            pred_confidences = outputs["confidences"]
            
            # Calculate metrics for this sample
            best_idx = torch.argmax(pred_confidences[0])
            pred_bbox = pred_bboxes[0, best_idx]
            pred_conf = pred_confidences[0, best_idx]
            
            # Calculate IoU
            iou = self._calculate_iou(pred_bbox, target_bbox[0])
            
            # Calculate other metrics
            precision, recall, f1 = self._calculate_prf1(
                pred_bbox, pred_conf, target_bbox[0], iou
            )
            
            accuracy = 1.0 if iou >= self.iou_threshold else 0.0
            
            return {
                "accuracy": accuracy,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confidence": pred_conf.item(),
            }
    
    def _calculate_iou(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> float:
        """Calculate IoU between two bounding boxes."""
        # Convert to [x1, y1, x2, y2] format
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_prf1(
        self,
        pred_bbox: torch.Tensor,
        pred_conf: torch.Tensor,
        target_bbox: torch.Tensor,
        iou: float,
    ) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        tp = 1 if iou >= self.iou_threshold else 0
        fp = 1 if iou < self.iou_threshold and pred_conf > 0.5 else 0
        fn = 1 if iou < self.iou_threshold and pred_conf <= 0.5 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1


def create_leaderboard(results: Dict[str, Dict[str, float]]) -> str:
    """Create a formatted leaderboard from evaluation results.
    
    Args:
        results: Dictionary mapping model names to their metrics
        
    Returns:
        Formatted leaderboard string
    """
    if not results:
        return "No results to display."
    
    # Get all metric names
    all_metrics = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    
    all_metrics = sorted(list(all_metrics))
    
    # Create header
    header = f"{'Model':<20}"
    for metric in all_metrics:
        header += f"{metric:<12}"
    
    # Create rows
    rows = [header]
    rows.append("-" * len(header))
    
    # Sort models by accuracy (if available)
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get("accuracy", 0.0),
        reverse=True
    )
    
    for model_name, metrics in sorted_models:
        row = f"{model_name:<20}"
        for metric in all_metrics:
            value = metrics.get(metric, 0.0)
            row += f"{value:<12.4f}"
        rows.append(row)
    
    return "\n".join(rows)
