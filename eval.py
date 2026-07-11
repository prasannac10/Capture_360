# eval.py
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import PanoramaDataset
from models.encoder import ImageEncoder
from models.aggregator import SetAggregator
from models.decoder import PanoramaDecoder
from models.panorama_model import PanoramaModel
from losses.geometry import geometry_loss
from losses.supervised import supervised_loss
from utils.checkpoint import load_checkpoint

# -------------------------------------------------
# Setup
# -------------------------------------------------
cfg = yaml.safe_load(open("config.yaml", "r"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert cfg["training"]["mode"] in ["supervised", "geometry_only"]

# -------------------------------------------------
# Dataset (VAL ONLY)
# -------------------------------------------------
val_dataset = PanoramaDataset(
    cfg["training"]["val_data"],
    has_gt=(cfg["training"]["mode"] == "supervised")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,                 # panoramas are heavy
    shuffle=False,
    collate_fn=lambda x: x[0]
)

# -------------------------------------------------
# Model (must EXACTLY match training)
# -------------------------------------------------
encoder = ImageEncoder(cfg["model"]["feature_dim"])
aggregator = SetAggregator(cfg["model"]["feature_dim"])
decoder = PanoramaDecoder(cfg["model"]["feature_dim"])

model = PanoramaModel(
    encoder,
    aggregator,
    decoder,
    pano_height=cfg["model"]["pano_height"],
    pano_width=cfg["model"]["pano_width"]
).to(device)

# -------------------------------------------------
# Load checkpoint
# -------------------------------------------------
ckpt_path = cfg["inference"]["checkpoint"]
load_checkpoint(model, ckpt_path, map_location=device)

model.eval()
print(f"[EVAL] Loaded checkpoint: {ckpt_path}")

# -------------------------------------------------
# Evaluation loop
# -------------------------------------------------
total_loss = 0.0
num_samples = 0

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        images = batch["images"].to(device)
        rotations = batch["rotations"].to(device)

        pano_pred = model(images, rotations)

        if cfg["training"]["mode"] == "geometry_only":
            loss = geometry_loss(pano_pred)
        else:
            pano_gt = batch["gt_panorama"].to(device)
            loss = supervised_loss(
                pano_pred,
                pano_gt,
                l1_weight=cfg["loss"]["supervised"]["l1_weight"]
            )

        total_loss += loss.item()
        num_samples += 1

# -------------------------------------------------
# Report
# -------------------------------------------------
avg_loss = total_loss / max(1, num_samples)

print("\n[EVAL COMPLETE]")
print(f"Samples evaluated : {num_samples}")
print(f"Average loss      : {avg_loss:.6f}")