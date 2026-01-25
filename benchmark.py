#!/usr/bin/env python3
"""
Benchmark script for comparing different referring expression comprehension models.

This script trains and evaluates multiple models to create a comprehensive
benchmark comparison.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.referring_expression import CLIPReferringExpressionModel
from src.data.dataset import ReferringExpressionDataModule, create_synthetic_dataset
from src.train.trainer import ReferringExpressionTrainer
from src.eval.metrics import ReferringExpressionEvaluator, create_leaderboard
from src.utils.core import get_device, set_seed, setup_logging, load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark referring expression comprehension models"
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
        default="benchmark_outputs",
        help="Directory to save benchmark results"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=["clip_referring_expression"],
        help="Models to benchmark"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
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
        "--create_synthetic",
        action="store_true",
        help="Create synthetic dataset for demo"
    )
    
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training, only evaluate existing checkpoints"
    )
    
    return parser.parse_args()


def main():
    """Main benchmark function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging("INFO")
    logger.info("Starting model benchmark")
    
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
    
    # Create synthetic dataset if requested
    if args.create_synthetic:
        logger.info("Creating synthetic dataset...")
        create_synthetic_dataset(args.data_dir, num_samples=1000)
    
    # Setup data module
    data_module = ReferringExpressionDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=224,
        max_length=128,
    )
    data_module.setup()
    
    logger.info(f"Train samples: {len(data_module.train_dataset)}")
    logger.info(f"Val samples: {len(data_module.val_dataset)}")
    logger.info(f"Test samples: {len(data_module.test_dataset)}")
    
    # Benchmark results
    benchmark_results = {}
    
    # Train and evaluate each model
    for model_name in args.models:
        logger.info(f"\n{'='*50}")
        logger.info(f"Benchmarking {model_name}")
        logger.info(f"{'='*50}")
        
        model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Initialize model
        if model_name == "clip_referring_expression":
            model = CLIPReferringExpressionModel()
        else:
            logger.warning(f"Model {model_name} not implemented yet, skipping...")
            continue
        
        model.to(device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training configuration
        config = OmegaConf.create({
            "num_epochs": args.epochs,
            "learning_rate": 1e-4,
            "weight_decay": 1e-4,
            "bbox_weight": 1.0,
            "confidence_weight": 1.0,
            "iou_weight": 0.5,
            "iou_threshold": 0.5,
            "output_dir": model_output_dir,
            "save_every": 10,
        })
        
        # Train model if not skipping training
        if not args.skip_training:
            logger.info(f"Training {model_name}...")
            
            trainer = ReferringExpressionTrainer(
                model=model,
                train_dataloader=data_module.train_dataloader(),
                val_dataloader=data_module.val_dataloader(),
                device=device,
                config=config,
            )
            
            trainer.train()
            
            # Load best model
            best_checkpoint = os.path.join(model_output_dir, "best_model.pth")
            if os.path.exists(best_checkpoint):
                checkpoint = torch.load(best_checkpoint, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded best model from {best_checkpoint}")
        else:
            # Try to load existing checkpoint
            checkpoint_path = os.path.join(model_output_dir, "best_model.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded existing checkpoint from {checkpoint_path}")
            else:
                logger.warning(f"No checkpoint found for {model_name}, skipping...")
                continue
        
        # Evaluate model
        logger.info(f"Evaluating {model_name}...")
        evaluator = ReferringExpressionEvaluator(iou_threshold=0.5)
        
        # Validation metrics
        val_metrics = evaluator.evaluate(
            model, data_module.val_dataloader(), device
        )
        
        # Test metrics
        test_metrics = evaluator.evaluate(
            model, data_module.test_dataloader(), device
        )
        
        # Store results
        benchmark_results[model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Validation - Accuracy: {val_metrics['accuracy']:.4f}, IoU: {val_metrics['mean_iou']:.4f}")
        logger.info(f"  Test - Accuracy: {test_metrics['accuracy']:.4f}, IoU: {test_metrics['mean_iou']:.4f}")
    
    # Create comprehensive leaderboard
    logger.info(f"\n{'='*50}")
    logger.info("BENCHMARK RESULTS")
    logger.info(f"{'='*50}")
    
    # Create leaderboard for test results
    test_results = {name: results["test"] for name, results in benchmark_results.items()}
    leaderboard = create_leaderboard(test_results)
    logger.info(f"\nTest Set Leaderboard:\n{leaderboard}")
    
    # Save results
    save_benchmark_results(benchmark_results, args.output_dir)
    
    # Generate comparison plots
    generate_comparison_plots(benchmark_results, args.output_dir)
    
    logger.info(f"\nBenchmark completed! Results saved to {args.output_dir}")


def save_benchmark_results(results: Dict[str, Dict], output_dir: str):
    """Save benchmark results to files."""
    import json
    
    # Save detailed results as JSON
    results_file = os.path.join(output_dir, "benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save results as CSV
    csv_data = []
    for model_name, model_results in results.items():
        row = {
            "Model": model_name,
            "Parameters": model_results["parameters"],
        }
        
        # Add validation metrics
        for metric, value in model_results["validation"].items():
            row[f"Val_{metric}"] = value
        
        # Add test metrics
        for metric, value in model_results["test"].items():
            row[f"Test_{metric}"] = value
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv_file = os.path.join(output_dir, "benchmark_results.csv")
    df.to_csv(csv_file, index=False)
    
    print(f"Results saved to {results_file} and {csv_file}")


def generate_comparison_plots(results: Dict[str, Dict], output_dir: str):
    """Generate comparison plots."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    models = list(results.keys())
    metrics = ["accuracy", "mean_iou", "precision", "recall", "f1"]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        val_scores = [results[model]["validation"][metric] for model in models]
        test_scores = [results[model]["test"][metric] for model in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[i].bar(x - width/2, val_scores, width, label='Validation', alpha=0.8)
        axes[i].bar(x + width/2, test_scores, width, label='Test', alpha=0.8)
        
        axes[i].set_title(f'{metric.replace("_", " ").title()}')
        axes[i].set_ylabel('Score')
        axes[i].set_xlabel('Model')
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(models, rotation=45)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Parameter count comparison
    param_counts = [results[model]["parameters"] for model in models]
    axes[5].bar(models, param_counts, alpha=0.8, color='orange')
    axes[5].set_title('Model Parameters')
    axes[5].set_ylabel('Parameter Count')
    axes[5].set_xlabel('Model')
    axes[5].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "benchmark_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plots saved to {os.path.join(output_dir, 'benchmark_comparison.png')}")


if __name__ == "__main__":
    main()
