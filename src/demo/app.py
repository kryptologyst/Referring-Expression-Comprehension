"""
Streamlit demo application for referring expression comprehension.

This module provides an interactive web interface for testing
referring expression comprehension models.
"""

import os
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..models.referring_expression import CLIPReferringExpressionModel
from ..utils.core import get_device, set_seed
from ..eval.metrics import ReferringExpressionEvaluator


class ReferringExpressionDemo:
    """Demo application for referring expression comprehension."""
    
    def __init__(self):
        """Initialize the demo application."""
        self.device = get_device()
        self.model = None
        self.evaluator = ReferringExpressionEvaluator()
        
        # Set random seed for reproducibility
        set_seed(42)
    
    def load_model(self, model_name: str = "clip_referring_expression") -> None:
        """Load the specified model.
        
        Args:
            model_name: Name of the model to load
        """
        try:
            if model_name == "clip_referring_expression":
                self.model = CLIPReferringExpressionModel()
            else:
                st.error(f"Model {model_name} not supported yet")
                return
            
            self.model.to(self.device)
            self.model.eval()
            st.success(f"Successfully loaded {model_name} model")
            
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
    
    def predict(
        self,
        image: Image.Image,
        text: str,
        return_attention: bool = False,
    ) -> Dict[str, any]:
        """Make prediction on image and text.
        
        Args:
            image: Input image
            text: Referring expression
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing predictions
        """
        if self.model is None:
            st.error("Model not loaded. Please load a model first.")
            return {}
        
        try:
            # Convert PIL image to tensor
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor, [text], return_attention=return_attention)
            
            # Get best prediction
            best_idx = torch.argmax(outputs["confidences"][0])
            best_bbox = outputs["bboxes"][0, best_idx]
            best_conf = outputs["confidences"][0, best_idx]
            
            # Convert normalized coordinates to pixel coordinates
            img_width, img_height = image.size
            bbox_pixels = [
                best_bbox[0].item() * img_width,
                best_bbox[1].item() * img_height,
                best_bbox[2].item() * img_width,
                best_bbox[3].item() * img_height,
            ]
            
            result = {
                "bbox": bbox_pixels,
                "confidence": best_conf.item(),
                "image_size": (img_width, img_height),
            }
            
            if return_attention and "attention_weights" in outputs:
                result["attention_weights"] = outputs["attention_weights"]
            
            return result
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return {}
    
    def visualize_prediction(
        self,
        image: Image.Image,
        prediction: Dict[str, any],
        show_confidence: bool = True,
    ) -> Image.Image:
        """Visualize prediction on image.
        
        Args:
            image: Input image
            prediction: Prediction results
            show_confidence: Whether to show confidence score
            
        Returns:
            Image with prediction visualization
        """
        if not prediction:
            return image
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(image)
        
        # Draw bounding box
        bbox = prediction["bbox"]
        x, y, w, h = bbox
        
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=3,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add confidence text
        if show_confidence:
            confidence = prediction["confidence"]
            ax.text(
                x, y - 10,
                f"Confidence: {confidence:.3f}",
                fontsize=12,
                color='red',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
            )
        
        ax.set_title("Referring Expression Comprehension Result", fontsize=14)
        ax.axis('off')
        
        # Convert to PIL Image
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return Image.fromarray(img_array)


def main():
    """Main demo application."""
    st.set_page_config(
        page_title="Referring Expression Comprehension",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Referring Expression Comprehension")
    st.markdown("""
    This demo showcases advanced referring expression comprehension models that can
    identify objects in images based on natural language descriptions.
    """)
    
    # Initialize demo
    if 'demo' not in st.session_state:
        st.session_state.demo = ReferringExpressionDemo()
    
    demo = st.session_state.demo
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["clip_referring_expression"],
        help="Choose the model architecture to use"
    )
    
    if st.sidebar.button("Load Model"):
        with st.spinner("Loading model..."):
            demo.load_model(model_name)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to analyze"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            # Use default image
            st.info("No image uploaded. Using default image.")
            # Create a simple default image
            image = Image.new('RGB', (224, 224), color='lightblue')
            st.image(image, caption="Default Image", use_column_width=True)
        
        # Text input
        referring_expression = st.text_input(
            "Referring Expression",
            value="the red object in the center",
            help="Describe the object you want to find in the image"
        )
        
        # Prediction options
        st.subheader("Options")
        show_attention = st.checkbox("Show Attention Weights", value=False)
        show_confidence = st.checkbox("Show Confidence Score", value=True)
    
    with col2:
        st.header("Prediction")
        
        if st.button("Analyze", type="primary"):
            if demo.model is None:
                st.error("Please load a model first using the sidebar.")
            else:
                with st.spinner("Analyzing image..."):
                    # Make prediction
                    prediction = demo.predict(
                        image,
                        referring_expression,
                        return_attention=show_attention
                    )
                    
                    if prediction:
                        # Display results
                        st.subheader("Results")
                        
                        col2_1, col2_2 = st.columns(2)
                        with col2_1:
                            st.metric("Confidence", f"{prediction['confidence']:.3f}")
                        
                        with col2_2:
                            bbox = prediction['bbox']
                            st.metric("Bounding Box", f"({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f})")
                        
                        # Visualize prediction
                        result_image = demo.visualize_prediction(
                            image,
                            prediction,
                            show_confidence=show_confidence
                        )
                        st.image(result_image, caption="Prediction Result", use_column_width=True)
                        
                        # Show attention weights if requested
                        if show_attention and "attention_weights" in prediction:
                            st.subheader("Attention Weights")
                            attention = prediction["attention_weights"]
                            st.write(f"Attention shape: {attention.shape}")
                            # You could visualize attention weights here
    
    # Examples section
    st.header("Examples")
    st.markdown("""
    Try these example referring expressions:
    - "the red ball on the table"
    - "the person wearing a blue shirt"
    - "the dog sitting on the grass"
    - "the car parked near the building"
    - "the bird flying in the sky"
    """)
    
    # Model information
    st.header("Model Information")
    st.markdown("""
    **CLIP Referring Expression Model**: This model uses CLIP's vision and text encoders
    to perform referring expression comprehension. It computes similarity between image
    regions and referring expressions to identify the described objects.
    
    **Features**:
    - Pre-trained CLIP backbone for robust vision-language understanding
    - Cross-attention mechanism for referring expression comprehension
    - Confidence scoring for prediction reliability
    - Support for various referring expression types
    """)


if __name__ == "__main__":
    main()
