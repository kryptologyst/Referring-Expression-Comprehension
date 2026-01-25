#!/usr/bin/env python3
"""
Main training script for referring expression comprehension.

This script provides a command-line interface for training and evaluating
referring expression comprehension models.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
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
        description="Train referring expression comprehension models"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
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
        default="outputs",
        help="Directory to save outputs"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="clip_referring_expression",
        choices=["clip_referring_expression", "mdetr", "lavt", "reftr"],
        help="Model architecture to use"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
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
        "--eval_only",
        action="store_true",
        help="Only evaluate, don't train"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging("INFO")
    logger.info("Starting referring expression comprehension training")
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        logger.warning(f"Configuration file {args.config} not found, using defaults")
        config = OmegaConf.create({})
    
    # Override config with command line arguments
    config.data_dir = args.data_dir
    config.output_dir = args.output_dir
    config.model.name = args.model
    config.data.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.lr
    
    # Setup device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
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
    
    # Initialize model
    if args.model == "clip_referring_expression":
        model = CLIPReferringExpressionModel()
    else:
        raise ValueError(f"Model {args.model} not implemented yet")
    
    model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = ReferringExpressionTrainer(
        model=model,
        train_dataloader=data_module.train_dataloader(),
        val_dataloader=data_module.val_dataloader(),
        device=device,
        config=config.training,
    )
    
    # Load checkpoint if provided
    if args.checkpoint:
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    # Train or evaluate
    if args.eval_only:
        logger.info("Running evaluation only...")
        
        # Evaluate on validation set
        val_metrics = trainer.validate()
        logger.info("Validation metrics:")
        for metric, value in val_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Evaluate on test set
        evaluator = ReferringExpressionEvaluator()
        test_metrics = evaluator.evaluate(
            model, data_module.test_dataloader(), device
        )
        logger.info("Test metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Create leaderboard
        results = {
            "Validation": val_metrics,
            "Test": test_metrics,
        }
        leaderboard = create_leaderboard(results)
        logger.info(f"\nLeaderboard:\n{leaderboard}")
        
    else:
        logger.info("Starting training...")
        trainer.train()
        
        # Final evaluation
        logger.info("Running final evaluation...")
        evaluator = ReferringExpressionEvaluator()
        
        # Evaluate on test set
        test_metrics = evaluator.evaluate(
            model, data_module.test_dataloader(), device
        )
        logger.info("Final test metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("Training/evaluation completed successfully!")


if __name__ == "__main__":
    main()
