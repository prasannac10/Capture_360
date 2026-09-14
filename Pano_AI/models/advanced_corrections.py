"""AI models for parallax correction, ghost removal, and overlap detection.

These models integrate into the existing Pano_AI pipeline architecture,
complementing the current correction models (glare, nadir_zenith, color).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.restoration_backbone import RestorationUNet, ConvBlock


class ParallaxCorrectionUNet(RestorationUNet):
    """Parallax correction U-Net for handling depth-based distortions.
    
    Inherits from RestorationUNet to maintain consistency with glare, nadir_zenith,
    and color correction models. Operates on the initial panorama to correct
    parallax artifacts caused by non-planar scenes.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 32):
        """
        Initialize parallax correction model.
        
        Args:
            in_channels: Input channels (3 for RGB)
            out_channels: Output channels (3 for RGB residual)
            base_channels: Base channel count for U-Net
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            mask_channels=0  # No explicit mask needed; learned from content
        )


class GhostRemovalUNet(RestorationUNet):
    """Ghost artifact removal U-Net for multi-exposure handling.
    
    Removes ghosting artifacts from moving objects or exposure inconsistencies
    in the panoramic blend. Uses the same backbone as other correction models
    for seamless integration into the pipeline.
    """
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 32):
        """
        Initialize ghost removal model.
        
        Args:
            in_channels: Input channels (3 for RGB)
            out_channels: Output channels (3 for RGB residual)
            base_channels: Base channel count for U-Net
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            mask_channels=1  # Optional ghost mask input
        )


class SeamBlendingUNet(nn.Module):
    """Learned seam-aware blending network.
    
    Learns optimal blending weights across seams by training on panorama quality metrics.
    Replaces or augments the classical multi-band blending approach.
    """
    
    def __init__(self, base_channels: int = 32):
        """
        Initialize seam blending network.
        
        Args:
            base_channels: Base channel count
        """
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        
        # Input: concatenation of two images in overlap region + seam edge map
        self.encoder = nn.Sequential(
            ConvBlock(6 + 1, c1),  # 6 channels for 2 images, 1 for seam edges
            nn.MaxPool2d(2),
            ConvBlock(c1, c2),
            nn.MaxPool2d(2),
            ConvBlock(c2, c3),
        )
        
        self.bottleneck = ConvBlock(c3, c3)
        
        self.decoder = nn.Sequential(
            ConvBlock(c3 * 2, c2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(c2 * 2, c1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(c1 * 2, c1),
        )
        
        # Output: blending weight map [0, 1]
        self.head = nn.Sequential(
            nn.Conv2d(c1, c1 // 2, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1 // 2, 1, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor, seam_edge: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute optimal blending weights.
        
        Args:
            img1, img2: Input images [B, 3, H, W]
            seam_edge: Seam edge map [B, 1, H, W]
            
        Returns:
            Blending weight map [B, 1, H, W] where 1 = take from img1, 0 = take from img2
        """
        combined = torch.cat([img1, img2, seam_edge], dim=1)
        
        e1 = self.encoder[0](combined)
        e2 = self.encoder[2](self.encoder[1](e1))
        e3 = self.encoder[4](self.encoder[3](e2))
        
        b = self.bottleneck(e3)
        
        d3 = self.decoder[0](torch.cat([
            F.interpolate(b, size=e3.shape[-2:], mode='bilinear', align_corners=False),
            e3
        ], dim=1))
        
        d2 = self.decoder[2](torch.cat([
            self.decoder[1](d3),
            e2
        ], dim=1))
        
        d1 = self.decoder[4](torch.cat([
            self.decoder[3](d2),
            e1
        ], dim=1))
        
        weight = self.head(d1)
        return weight


class OverlapDetectionUNet(nn.Module):
    """Learned overlap region detection network.
    
    Predicts overlap confidence and boundaries between image pairs.
    Complements classical feature matching for more robust overlap detection.
    """
    
    def __init__(self, base_channels: int = 32):
        """
        Initialize overlap detection network.
        
        Args:
            base_channels: Base channel count
        """
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        
        # Siamese encoder for two images
        self.shared_encoder = nn.Sequential(
            ConvBlock(3, c1),
            nn.MaxPool2d(2),
            ConvBlock(c1, c2),
            nn.MaxPool2d(2),
            ConvBlock(c2, c3),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(c3 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        # Output heads
        self.overlap_prob_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.overlap_percentage_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output [0, 1], multiply by 100 for percentage
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> dict:
        """
        Forward pass for overlap detection.
        
        Args:
            img1, img2: Input images [B, 3, H, W]
            
        Returns:
            Dictionary with:
                - overlap_prob: Probability of overlap [B, 1]
                - overlap_percentage: Estimated overlap % [B, 1]
                - confidence: Confidence score [B, 1]
        """
        feat1 = self.shared_encoder(img1).view(img1.size(0), -1)
        feat2 = self.shared_encoder(img2).view(img2.size(0), -1)
        
        combined = torch.cat([feat1, feat2, torch.abs(feat1 - feat2)], dim=1)
        shared = self.classifier(combined)
        
        return {
            'overlap_prob': self.overlap_prob_head(shared),
            'overlap_percentage': self.overlap_percentage_head(shared) * 100.0,
            'confidence': self.confidence_head(shared)
        }


class ParallaxCorrectionDetector(nn.Module):
    """Parallax distortion detection network.
    
    Estimates parallax magnitude and direction in overlap regions.
    Outputs parallax flow field for warping correction.
    """
    
    def __init__(self, base_channels: int = 32):
        """
        Initialize parallax detection network.
        
        Args:
            base_channels: Base channel count
        """
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        
        # Encode difference image
        self.encoder = nn.Sequential(
            ConvBlock(3, c1),  # Difference image: |img1 - img2|
            nn.MaxPool2d(2),
            ConvBlock(c1, c2),
            nn.MaxPool2d(2),
            ConvBlock(c2, c3),
        )
        
        self.bottleneck = ConvBlock(c3, c3)
        
        # Decoder for parallax flow
        self.decoder = nn.Sequential(
            ConvBlock(c3 * 2, c2),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBlock(c2 * 2, c1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        
        # Output: 2D parallax flow field
        self.head = nn.Conv2d(c1, 2, 3, padding=1)  # [dx, dy] per pixel
    
    def forward(self, img1: torch.Tensor, img2: torch.Tensor, overlap_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass to predict parallax flow.
        
        Args:
            img1, img2: Input images [B, 3, H, W]
            overlap_mask: Optional overlap region mask [B, 1, H, W]
            
        Returns:
            Parallax flow field [B, 2, H, W] representing [dx, dy] shifts
        """
        # Compute difference image
        diff = torch.abs(img1 - img2)
        
        e1 = self.encoder[0](diff)
        e2 = self.encoder[2](self.encoder[1](e1))
        e3 = self.encoder[4](self.encoder[3](e2))
        
        b = self.bottleneck(e3)
        
        d2 = self.decoder[0](torch.cat([
            F.interpolate(b, size=e3.shape[-2:], mode='bilinear', align_corners=False),
            e3
        ], dim=1))
        
        d1 = self.decoder[2](torch.cat([
            self.decoder[1](d2),
            e2
        ], dim=1))
        
        flow = self.head(d1)
        
        # Mask out non-overlap regions
        if overlap_mask is not None:
            flow = flow * overlap_mask.to(flow.dtype)
        
        return flow
