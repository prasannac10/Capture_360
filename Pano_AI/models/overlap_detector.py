"""AI-based overlap detection using deep learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import numpy as np


class OverlapDetectorNet(nn.Module):
    """Deep learning model for detecting and characterizing overlaps between views."""

    def __init__(self, feature_dim: int = 128):
        """
        Initialize overlap detection network.
        
        Args:
            feature_dim: Feature dimension for embeddings
        """
        super().__init__()
        self.feature_dim = feature_dim

        # Siamese feature encoder (shared weights)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual blocks
            self._make_residual_block(64, 64, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            self._make_residual_block(64, 128, 3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            self._make_residual_block(128, 256, 3),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Overlap classification head
        self.overlap_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Overlap percentage regression head
        self.overlap_regressor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output range [0, 1] representing 0-100%
        )

        # Alignment confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def _make_residual_block(self, in_channels: int, out_channels: int, num_blocks: int):
        """Create a residual block."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=1 if in_channels == out_channels else 2))
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for overlap detection.
        
        Args:
            img1, img2: Input images [B, 3, H, W]
            
        Returns:
            Dictionary with:
                - overlap_prob: Probability of overlap [B, 1]
                - overlap_percentage: Estimated overlap percentage [B, 1]
                - confidence: Confidence score [B, 1]
        """
        # Encode both images
        feat1 = self.encoder(img1)
        feat2 = self.encoder(img2)
        
        # Flatten features
        feat1_flat = feat1.view(feat1.size(0), -1)
        feat2_flat = feat2.view(feat2.size(0), -1)
        
        # Concatenate features for pairwise comparison
        combined = torch.cat([feat1_flat, feat2_flat, torch.abs(feat1_flat - feat2_flat)], dim=1)
        
        # Predict overlap properties
        overlap_prob = self.overlap_classifier(combined)
        overlap_pct = self.overlap_regressor(combined) * 100  # Scale to 0-100%
        confidence = self.confidence_head(combined)
        
        return {
            'overlap_prob': overlap_prob,
            'overlap_percentage': overlap_pct,
            'confidence': confidence
        }


class ResidualBlock(nn.Module):
    """Basic residual block."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class OverlapDetectorAI:
    """Wrapper for AI-based overlap detection."""
    
    def __init__(self, model: OverlapDetectorNet, device: str = "cpu", threshold: float = 0.5):
        """
        Initialize AI overlap detector.
        
        Args:
            model: Trained OverlapDetectorNet model
            device: Device to run model on ("cpu" or "cuda")
            threshold: Overlap probability threshold
        """
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold
        self.model.eval()
    
    def detect_overlaps_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 4
    ) -> Dict[Tuple[int, int], Dict]:
        """
        Detect overlaps between all image pairs using AI model.
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
            
        Returns:
            Dictionary mapping (i, j) -> overlap results
        """
        n = len(images)
        overlaps = {}
        
        with torch.no_grad():
            for i in range(n):
                for j in range(i + 1, n):
                    # Convert images to tensors
                    img1_tensor = self._image_to_tensor(images[i])
                    img2_tensor = self._image_to_tensor(images[j])
                    
                    # Predict overlap
                    results = self.model(img1_tensor, img2_tensor)
                    
                    overlap_prob = results['overlap_prob'].item()
                    overlap_pct = results['overlap_percentage'].item()
                    confidence = results['confidence'].item()
                    
                    if overlap_prob > self.threshold:
                        overlaps[(i, j)] = {
                            'overlap_prob': overlap_prob,
                            'overlap_percentage': overlap_pct,
                            'confidence': confidence,
                            'is_overlapping': True
                        }
                    else:
                        overlaps[(i, j)] = {
                            'overlap_prob': overlap_prob,
                            'overlap_percentage': 0.0,
                            'confidence': confidence,
                            'is_overlapping': False
                        }
        
        return overlaps
    
    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy image to tensor."""
        # Resize to 256x256 for model input
        import cv2
        image_resized = cv2.resize(image, (256, 256))
        
        # Convert BGR to RGB and normalize
        if len(image_resized.shape) == 3:
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        image_normalized = image_rgb.astype(np.float32) / 255.0
        
        # Convert to tensor [1, C, H, W]
        tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
