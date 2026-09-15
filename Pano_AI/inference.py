import argparse
import os
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from data.collate import panorama_collate_fn
from data.dataset import PanoramaDataset
from models.aggregator import SetAggregator
from models.decoder import PanoramaDecoder
from models.encoder import ImageEncoder
from models.panorama_model import PanoramaModel
from utils.checkpoint import load_checkpoint
from utils.ema import EMA


def pano_angles(h, w):
    ys, xs = torch.meshgrid(torch.linspace(-0.5, 0.5, h), torch.linspace(-1.0, 1.0, w), indexing="ij")
    return xs * 180.0, -ys * 180.0


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config.yaml"); args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle: cfg = yaml.safe_load(handle)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PanoramaDataset(cfg["training"]["val_data"], has_gt=(cfg["training"]["mode"] == "supervised"))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=panorama_collate_fn)
    model = PanoramaModel(ImageEncoder(cfg["model"]["feature_dim"]), SetAggregator(cfg["model"]["feature_dim"]), PanoramaDecoder(cfg["model"]["feature_dim"]), cfg["model"]["pano_height"], cfg["model"]["pano_width"]).to(device)
    ema = EMA(model, cfg["training"]["ema_decay"]) if cfg["training"].get("use_ema", False) else None
    load_checkpoint(cfg["inference"]["checkpoint"], model, device=device, ema=ema, use_ema=ema is not None)
    model.eval(); out_dir = cfg["inference"]["output_dir"]; os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for index, batch in enumerate(loader):
            pano = model(batch["images"].to(device), batch["rotations"].to(device), batch["mask"].to(device))[0].clamp(0, 1)
            if cfg["inference"].get("save_intermediate", False): torch.save(model.last_spherical[0].cpu(), os.path.join(out_dir, f"spherical_{index:04d}.pt"))
            if cfg["inference"].get("save_metadata", False):
                yaw, pitch = pano_angles(pano.shape[1], pano.shape[2]); torch.save({"yaw": yaw, "pitch": pitch}, os.path.join(out_dir, f"pano_angles_{index:04d}.pt"))
            if cfg["inference"].get("save_outputs", False):
                arr = (pano.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8); Image.fromarray(arr).save(os.path.join(out_dir, f"pano_{index:04d}.png"))
                arr16 = (pano.permute(1, 2, 0).cpu().numpy() * 65535).round().astype(np.uint16); Image.fromarray(arr16).save(os.path.join(out_dir, f"pano_{index:04d}.tiff"), compression="tiff_deflate")


if __name__ == "__main__": main()
