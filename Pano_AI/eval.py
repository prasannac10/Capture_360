import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.collate import panorama_collate_fn
from data.dataset import PanoramaDataset
from losses.geometry import geometry_loss
from losses.supervised import supervised_loss
from models.aggregator import SetAggregator
from models.decoder import PanoramaDecoder
from models.encoder import ImageEncoder
from models.panorama_model import PanoramaModel
from utils.checkpoint import load_checkpoint
from utils.ema import EMA


def build_model(cfg):
    return PanoramaModel(ImageEncoder(cfg["model"]["feature_dim"]), SetAggregator(cfg["model"]["feature_dim"]), PanoramaDecoder(cfg["model"]["feature_dim"]), cfg["model"]["pano_height"], cfg["model"]["pano_width"])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default="config.yaml"); args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle: cfg = yaml.safe_load(handle)
    mode = cfg["training"]["mode"]; device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PanoramaDataset(cfg["training"]["val_data"], has_gt=(mode == "supervised"))
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=panorama_collate_fn)
    model = build_model(cfg).to(device)
    ema = EMA(model, cfg["training"]["ema_decay"]) if cfg["training"].get("use_ema", False) else None
    load_checkpoint(cfg["inference"]["checkpoint"], model, device=device, ema=ema, use_ema=ema is not None)
    model.eval(); total = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images, rotations, mask = batch["images"].to(device), batch["rotations"].to(device), batch["mask"].to(device)
            pred = model(images, rotations, mask)
            if mode == "supervised":
                loss = supervised_loss(pred, batch["gt_panorama"].to(device), cfg["loss"]["supervised"].get("l1_weight", 1.0), cfg["loss"]["supervised"].get("ssim_weight", 0.0))
            else:
                loss = geometry_loss(pred, images, rotations, mask, cfg["loss"]["geometry"].get("smoothness_weight", 1.0))
            total += loss.item()
    print(f"Samples evaluated : {len(dataset)}")
    print(f"Average loss      : {total / max(1, len(loader)):.6f}")


if __name__ == "__main__": main()
