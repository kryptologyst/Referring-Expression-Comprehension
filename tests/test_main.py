"""
Test suite for referring expression comprehension project.

This module contains unit tests for all major components.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import torch
import numpy as np
from PIL import Image

# Add src to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.referring_expression import CLIPReferringExpressionModel
from src.data.dataset import RefCOCODataset, ReferringExpressionDataModule
from src.eval.metrics import ReferringExpressionMetrics, ReferringExpressionEvaluator
from src.train.trainer import ReferringExpressionLoss
from src.utils.core import get_device, set_seed, AverageMeter


class TestCLIPReferringExpressionModel(unittest.TestCase):
    """Test CLIP referring expression model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = CLIPReferringExpressionModel()
        self.device = torch.device("cpu")
        self.model.to(self.device)
    
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.model.clip_model)
        self.assertIsNotNone(self.model.processor)
    
    def test_forward_pass(self):
        """Test forward pass."""
        # Create dummy input
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        texts = ["the red ball", "the blue car"]
        
        # Forward pass
        outputs = self.model(images, texts)
        
        # Check output structure
        self.assertIn("bboxes", outputs)
        self.assertIn("confidences", outputs)
        self.assertIn("image_features", outputs)
        self.assertIn("text_features", outputs)
        
        # Check output shapes
        self.assertEqual(outputs["bboxes"].shape, (batch_size, self.model.num_queries, 4))
        self.assertEqual(outputs["confidences"].shape, (batch_size, self.model.num_queries))
        self.assertEqual(outputs["image_features"].shape, (batch_size, self.model.hidden_dim))
        self.assertEqual(outputs["text_features"].shape, (batch_size, self.model.hidden_dim))


class TestRefCOCODataset(unittest.TestCase):
    """Test RefCOCO dataset."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset = RefCOCODataset(
            data_dir=self.temp_dir,
            split="train",
            max_length=128,
            image_size=224,
            augment=False,
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_length(self):
        """Test dataset length."""
        self.assertGreater(len(self.dataset), 0)
    
    def test_dataset_item(self):
        """Test dataset item structure."""
        item = self.dataset[0]
        
        self.assertIn("image", item)
        self.assertIn("text", item)
        self.assertIn("bbox", item)
        self.assertIn("image_id", item)
        self.assertIn("category", item)
        
        # Check types
        self.assertIsInstance(item["image"], torch.Tensor)
        self.assertIsInstance(item["text"], str)
        self.assertIsInstance(item["bbox"], torch.Tensor)
        self.assertIsInstance(item["image_id"], str)
        self.assertIsInstance(item["category"], str)
        
        # Check shapes
        self.assertEqual(item["image"].shape, (3, 224, 224))
        self.assertEqual(item["bbox"].shape, (4,))


class TestReferringExpressionMetrics(unittest.TestCase):
    """Test referring expression metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = ReferringExpressionMetrics(iou_threshold=0.5)
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        self.assertEqual(self.metrics.iou_threshold, 0.5)
        self.assertEqual(self.metrics.total_samples, 0)
        self.assertEqual(self.metrics.correct_predictions, 0)
    
    def test_metrics_update(self):
        """Test metrics update."""
        batch_size = 2
        num_queries = 10
        
        pred_bboxes = torch.randn(batch_size, num_queries, 4)
        pred_confidences = torch.rand(batch_size, num_queries)
        target_bboxes = torch.randn(batch_size, 4)
        
        self.metrics.update(pred_bboxes, pred_confidences, target_bboxes)
        
        self.assertEqual(self.metrics.total_samples, batch_size)
    
    def test_metrics_compute(self):
        """Test metrics computation."""
        # Add some dummy data
        batch_size = 2
        num_queries = 10
        
        pred_bboxes = torch.randn(batch_size, num_queries, 4)
        pred_confidences = torch.rand(batch_size, num_queries)
        target_bboxes = torch.randn(batch_size, 4)
        
        self.metrics.update(pred_bboxes, pred_confidences, target_bboxes)
        
        results = self.metrics.compute()
        
        self.assertIn("accuracy", results)
        self.assertIn("mean_iou", results)
        self.assertIn("precision", results)
        self.assertIn("recall", results)
        self.assertIn("f1", results)
        
        # Check value ranges
        self.assertGreaterEqual(results["accuracy"], 0.0)
        self.assertLessEqual(results["accuracy"], 1.0)
        self.assertGreaterEqual(results["mean_iou"], 0.0)
        self.assertLessEqual(results["mean_iou"], 1.0)


class TestReferringExpressionLoss(unittest.TestCase):
    """Test referring expression loss function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss_fn = ReferringExpressionLoss()
    
    def test_loss_computation(self):
        """Test loss computation."""
        batch_size = 2
        num_queries = 10
        
        pred_bboxes = torch.randn(batch_size, num_queries, 4)
        pred_confidences = torch.rand(batch_size, num_queries)
        target_bboxes = torch.randn(batch_size, 4)
        
        loss_dict = self.loss_fn(pred_bboxes, pred_confidences, target_bboxes)
        
        self.assertIn("total_loss", loss_dict)
        self.assertIn("bbox_loss", loss_dict)
        self.assertIn("confidence_loss", loss_dict)
        self.assertIn("iou_loss", loss_dict)
        
        # Check loss values are non-negative
        for loss_name, loss_value in loss_dict.items():
            self.assertGreaterEqual(loss_value.item(), 0.0)


class TestCoreUtils(unittest.TestCase):
    """Test core utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        self.assertIsInstance(device, torch.device)
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't raise an error
        self.assertTrue(True)
    
    def test_average_meter(self):
        """Test average meter."""
        meter = AverageMeter()
        
        # Test reset
        meter.reset()
        self.assertEqual(meter.val, 0.0)
        self.assertEqual(meter.avg, 0.0)
        self.assertEqual(meter.sum, 0.0)
        self.assertEqual(meter.count, 0)
        
        # Test update
        meter.update(1.0, 1)
        self.assertEqual(meter.val, 1.0)
        self.assertEqual(meter.avg, 1.0)
        self.assertEqual(meter.sum, 1.0)
        self.assertEqual(meter.count, 1)
        
        # Test multiple updates
        meter.update(2.0, 2)
        self.assertEqual(meter.val, 2.0)
        self.assertEqual(meter.avg, 5.0 / 3.0)  # (1*1 + 2*2) / 3
        self.assertEqual(meter.sum, 5.0)
        self.assertEqual(meter.count, 3)


class TestDataModule(unittest.TestCase):
    """Test data module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_module = ReferringExpressionDataModule(
            data_dir=self.temp_dir,
            batch_size=4,
            num_workers=0,  # Use 0 workers for testing
            image_size=224,
            max_length=128,
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_data_module_setup(self):
        """Test data module setup."""
        self.data_module.setup()
        
        self.assertIsNotNone(self.data_module.train_dataset)
        self.assertIsNotNone(self.data_module.val_dataset)
        self.assertIsNotNone(self.data_module.test_dataset)
    
    def test_dataloaders(self):
        """Test dataloaders."""
        self.data_module.setup()
        
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.val_dataloader()
        test_loader = self.data_module.test_dataloader()
        
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)
        
        # Test batch structure
        for loader in [train_loader, val_loader, test_loader]:
            batch = next(iter(loader))
            self.assertIn("images", batch)
            self.assertIn("texts", batch)
            self.assertIn("bboxes", batch)
            self.assertIn("image_ids", batch)
            self.assertIn("categories", batch)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
