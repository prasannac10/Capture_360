import yaml
import torch
from torch.utils.data import DataLoader
import os
import torchvision.utils as vutils
from data.dataset import PanoramaDataset
from models.encoder import ImageEncoder
from models.aggregator import SetAggregator
from models.decoder import PanoramaDecoder
from models.model import PanoramaModel
from losses.geometry import geometry_loss
from losses.supervised import supervised_loss
from utils.checkpoint import load_ckpt
from PIL import Image
import numpy as np


def pano_yaw_pitch(h, w):
    ys, xs = torch.meshgrid(
        torch.linspace(-0.5, 0.5, h),
        torch.linspace(-1.0, 1.0, w),
        indexing="ij"
    )
    yaw = xs * 180.0       # [-180, 180]
    pitch = -ys * 180.0    # [-90, 90]
    return yaw, pitch

# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------
cfg = yaml.safe_load(open("config.yaml"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# Dataset (validation / test)
# ---------------------------------------------------------
dataset = PanoramaDataset(
    "data/val",
    has_gt=(cfg["training"]["mode"] == "supervised")
)

loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])

# ---------------------------------------------------------
# Model (must match training exactly)
# ---------------------------------------------------------
encoder = ImageEncoder(cfg["model"]["feature_dim"])
aggregator = SetAggregator(cfg["model"]["feature_dim"])
decoder = PanoramaDecoder(cfg["model"]["feature_dim"])

model = PanoramaModel(
    encoder,
    aggregator,
    decoder,
    cfg["model"]["pano_height"],
    cfg["model"]["pano_width"]
).to(device)

# ---------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------
ckpt_path = cfg.get("inference", {}).get("checkpoint", "model.pt")
load_ckpt(model, ckpt_path)

model.eval()
print(f"Loaded checkpoint: {ckpt_path}")

# ---------------------------------------------------------
# Inference loop
# ---------------------------------------------------------
total_loss = 0.0
num_samples = 0

with torch.no_grad():
    for batch in loader:
        images = batch["images"].to(device)
        rotations = batch["rotations"].to(device)

        pano = model(images, rotations)

        if cfg["inference"].get("save_intermediate", False):
            torch.save(
                model.last_spherical.cpu(),
                os.path.join(out_dir, f"spherical_{num_samples:04d}.pt")
            )
        
        if cfg["inference"].get("save_metadata", False):
            yaw, pitch = pano_yaw_pitch(
                cfg["model"]["pano_height"],
                cfg["model"]["pano_width"]
            )

            torch.save(
                {"yaw": yaw, "pitch": pitch},
                os.path.join(out_dir, "pano_angles.pt")
            )

        if cfg["inference"].get("save_outputs", False):
            out_dir = cfg["inference"]["output_dir"]
            os.makedirs(out_dir, exist_ok=True)

            # pano: [3, H, W] or [1,3,H,W]
            pano_img = pano.clamp(0, 1)

            out_path = os.path.join(out_dir, f"pano_{num_samples:04d}.png")
            vutils.save_image(pano_img, out_path)

            pano_np = (
                pano_img.permute(1,2,0).cpu().numpy() * 65535
            ).astype(np.uint16)

            Image.fromarray(pano_np).save(
                out_path.replace(".png", ".tiff"),
                compression="tiff_deflate"
            )

        if cfg["training"]["mode"] == "geometry_only":
            loss = geometry_loss(pano)
        else:
            gt = batch["gt_panorama"].to(device)
            loss = supervised_loss(
                pano,
                gt,
                cfg["loss"]["supervised"]["l1_weight"]
            )

        total_loss += loss.item()
        num_samples += 1

        # -------------------------------------------------
        # Optional: qualitative hook (first sample only)
        # -------------------------------------------------
        if num_samples == 1:
            print("Panorama output stats:")
            print("  shape:", pano.shape)
            print("  min/max:", pano.min().item(), pano.max().item())

avg_loss = total_loss / max(1, num_samples)

print(f"\nValidation complete")
print(f"Average loss: {avg_loss:.4f}")