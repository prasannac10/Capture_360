import yaml
import torch
from torch.utils.data import DataLoader

from data.dataset import PanoramaDataset
from models.encoder import ImageEncoder
from models.aggregator import SetAggregator
from models.decoder import PanoramaDecoder
from models.panorama_model import PanoramaModel
from losses.geometry import geometry_loss
from losses.supervised import supervised_loss
from utils.ema import EMA
from utils.checkpoint import *

cfg = yaml.safe_load(open("config.yaml"))

dataset = PanoramaDataset(
    root = cfg["training"]["training_data"],
    has_gt=(cfg["training"]["mode"] == "supervised")
)

loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x[0])

encoder = ImageEncoder(cfg["model"]["feature_dim"])
aggregator = SetAggregator(cfg["model"]["feature_dim"])
decoder = PanoramaDecoder(cfg["model"]["feature_dim"])

model = PanoramaModel(
    encoder, aggregator, decoder,
    cfg["model"]["pano_height"],
    cfg["model"]["pano_width"]
)

opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

ema = EMA(model, cfg["training"]["ema_decay"]) if cfg["training"]["use_ema"] else None

for epoch in range(cfg["training"]["epochs"]):
    for batch in loader:
        pano = model(batch["images"], batch["rotations"])

        if cfg["training"]["mode"] == "geometry_only":
            loss = geometry_loss(pano)
        else:
            loss = supervised_loss(
                pano, batch["gt_panorama"],
                cfg["loss"]["supervised"]["l1_weight"]
            )

        opt.zero_grad()
        loss.backward()
        opt.step()

        if ema:
            ema.update(model)

    print(f"epoch {epoch} | loss {loss.item():.4f}")

save_checkpoint(cfg["training"]["model_name"], model)
