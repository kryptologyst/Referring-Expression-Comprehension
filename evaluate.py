#!/usr/bin/env python3
"""
Evaluation script for referring expression comprehension models.

This script provides comprehensive evaluation of trained models
with detailed metrics and visualization.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.referring_expression import CLIPReferringExpressionModel
from src.data.dataset import ReferringExpressionDataModule
from src.eval.metrics import ReferringExpressionEvaluator, create_leaderboard
from src.utils.core import get_device, set_seed, setup_logging, load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate referring expression comprehension models"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing dataset"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_outputs",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots"
    )
    
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save prediction results"
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging("INFO")
    logger.info("Starting model evaluation")
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup data module
    data_module = ReferringExpressionDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=224,
        max_length=128,
    )
    data_module.setup()
    
    logger.info(f"Test samples: {len(data_module.test_dataset)}")
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint.get("config", {})
    
    # Initialize model based on config
    model_name = config.get("model", {}).get("name", "clip_referring_expression")
    
    if model_name == "clip_referring_expression":
        model = CLIPReferringExpressionModel()
    else:
        raise ValueError(f"Model {model_name} not implemented yet")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {args.checkpoint}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize evaluator
    evaluator = ReferringExpressionEvaluator(iou_threshold=0.5)
    
    # Run evaluation
    logger.info("Running evaluation on test set...")
    test_metrics = evaluator.evaluate(
        model, data_module.test_dataloader(), device
    )
    
    # Print results
    logger.info("Test Results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Create leaderboard
    results = {"Test": test_metrics}
    leaderboard = create_leaderboard(results)
    logger.info(f"\nLeaderboard:\n{leaderboard}")
    
    # Save results
    results_file = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write("Referring Expression Comprehension Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test samples: {len(data_module.test_dataset)}\n\n")
        f.write("Metrics:\n")
        for metric, value in test_metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write(f"\nLeaderboard:\n{leaderboard}\n")
    
    logger.info(f"Results saved to {results_file}")
    
    # Generate visualizations if requested
    if args.visualize:
        logger.info("Generating visualization plots...")
        generate_visualizations(model, data_module.test_dataset, device, args.output_dir)
    
    # Save predictions if requested
    if args.save_predictions:
        logger.info("Saving prediction results...")
        save_predictions(model, data_module.test_dataset, device, args.output_dir)
    
    logger.info("Evaluation completed successfully!")


def generate_visualizations(model, dataset, device, output_dir):
    """Generate visualization plots."""
    import matplotlib.pyplot as plt
    
    # Sample a few examples for visualization
    num_samples = min(5, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        # Get prediction
        image = sample["image"].unsqueeze(0).to(device)
        text = sample["text"]
        
        with torch.no_grad():
            outputs = model(image, [text])
            best_idx = torch.argmax(outputs["confidences"][0])
            pred_bbox = outputs["bboxes"][0, best_idx]
            pred_conf = outputs["confidences"][0, best_idx]
        
        # Convert to pixel coordinates
        img_width, img_height = 224, 224  # Assuming square images
        pred_bbox_pixels = [
            pred_bbox[0].item() * img_width,
            pred_bbox[1].item() * img_height,
            pred_bbox[2].item() * img_width,
            pred_bbox[3].item() * img_height,
        ]
        
        target_bbox_pixels = [
            sample["bbox"][0].item() * img_width,
            sample["bbox"][1].item() * img_height,
            sample["bbox"][2].item() * img_width,
            sample["bbox"][3].item() * img_height,
        ]
        
        # Plot original image
        axes[0, i].imshow(image[0].cpu().permute(1, 2, 0))
        axes[0, i].set_title(f"Original: {text[:20]}...")
        axes[0, i].axis('off')
        
        # Plot prediction
        axes[1, i].imshow(image[0].cpu().permute(1, 2, 0))
        
        # Draw ground truth box
        from matplotlib.patches import Rectangle
        gt_rect = Rectangle(
            (target_bbox_pixels[0], target_bbox_pixels[1]),
            target_bbox_pixels[2], target_bbox_pixels[3],
            linewidth=2, edgecolor='green', facecolor='none', label='Ground Truth'
        )
        axes[1, i].add_patch(gt_rect)
        
        # Draw prediction box
        pred_rect = Rectangle(
            (pred_bbox_pixels[0], pred_bbox_pixels[1]),
            pred_bbox_pixels[2], pred_bbox_pixels[3],
            linewidth=2, edgecolor='red', facecolor='none', label='Prediction'
        )
        axes[1, i].add_patch(pred_rect)
        
        axes[1, i].set_title(f"Prediction (Conf: {pred_conf:.3f})")
        axes[1, i].axis('off')
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predictions_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {os.path.join(output_dir, 'predictions_visualization.png')}")


def save_predictions(model, dataset, device, output_dir):
    """Save prediction results to file."""
    import json
    
    predictions = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        # Get prediction
        image = sample["image"].unsqueeze(0).to(device)
        text = sample["text"]
        
        with torch.no_grad():
            outputs = model(image, [text])
            best_idx = torch.argmax(outputs["confidences"][0])
            pred_bbox = outputs["bboxes"][0, best_idx]
            pred_conf = outputs["confidences"][0, best_idx]
        
        prediction = {
            "image_id": sample["image_id"],
            "text": text,
            "predicted_bbox": pred_bbox.cpu().tolist(),
            "predicted_confidence": pred_conf.item(),
            "ground_truth_bbox": sample["bbox"].tolist(),
            "category": sample["category"],
        }
        
        predictions.append(prediction)
    
    # Save predictions
    predictions_file = os.path.join(output_dir, "predictions.json")
    with open(predictions_file, "w") as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to {predictions_file}")


if __name__ == "__main__":
    main()
