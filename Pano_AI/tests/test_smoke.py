import argparse
import sys
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader, Subset
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from data.collate import panorama_collate_fn
from data.dataset import PanoramaDataset
from losses.supervised import supervised_loss
from models.aggregator import SetAggregator
from models.decoder import PanoramaDecoder
from models.encoder import ImageEncoder
from models.panorama_model import PanoramaModel


def main():
    parser = argparse.ArgumentParser(description="Real-data forward/backward smoke test"); parser.add_argument("--config", default=str(ROOT / "config.yaml")); parser.add_argument("--samples", type=int, default=3); args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle: cfg = yaml.safe_load(handle)
    dataset = PanoramaDataset(cfg["training"]["training_data"], has_gt=True)
    count = min(args.samples, len(dataset))
    if count < 2: raise RuntimeError("Smoke test requires at least 2 real scene sets")
    loader = DataLoader(Subset(dataset, range(count)), batch_size=count, collate_fn=panorama_collate_fn)
    batch = next(iter(loader))
    model = PanoramaModel(ImageEncoder(cfg["model"]["feature_dim"]), SetAggregator(cfg["model"]["feature_dim"]), PanoramaDecoder(cfg["model"]["feature_dim"]), cfg["model"]["pano_height"], cfg["model"]["pano_width"])
    pred = model(batch["images"], batch["rotations"], batch["mask"])
    assert tuple(pred.shape) == (count, 3, 256, 512), pred.shape
    loss = supervised_loss(pred, batch["gt_panorama"]); loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    print(f"SMOKE PASS: {count} real scenes, output={tuple(pred.shape)}, loss={loss.item():.6f}")


if __name__ == "__main__": main()
