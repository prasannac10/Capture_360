import argparse
import json
import os
import torch
import yaml
from models.aggregator import SetAggregator
from models.decoder import PanoramaDecoder
from models.encoder import ImageEncoder
from models.panorama_model import PanoramaModel
from utils.checkpoint import load_checkpoint
from utils.ema import EMA

INPUT_N = 6; INPUT_H = 224; INPUT_W = 224


def build_model(cfg):
    return PanoramaModel(ImageEncoder(cfg["model"]["feature_dim"]), SetAggregator(cfg["model"]["feature_dim"]), PanoramaDecoder(cfg["model"]["feature_dim"]), cfg["model"]["pano_height"], cfg["model"]["pano_width"])


def main():
    parser = argparse.ArgumentParser(description="Export PanoramaModel to ONNX for Android.")
    parser.add_argument("--config", default="config.yaml"); parser.add_argument("--checkpoint", default=None); parser.add_argument("--output", default="artifacts/pano_model.onnx"); parser.add_argument("--num-frames", type=int, default=INPUT_N); args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle: cfg = yaml.safe_load(handle)
    model = build_model(cfg).eval()
    ema = EMA(model, cfg["training"]["ema_decay"]) if cfg["training"].get("use_ema", False) else None
    checkpoint = args.checkpoint or os.path.join(cfg["training"]["model_path"], cfg["training"]["model_name"])
    load_checkpoint(checkpoint, model, device="cpu", ema=ema, use_ema=ema is not None)
    images = torch.zeros(1, args.num_frames, 3, INPUT_H, INPUT_W)
    rotations = torch.eye(3).view(1, 1, 3, 3).repeat(1, args.num_frames, 1, 1)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.onnx.export(model, (images, rotations), args.output, input_names=["images", "rotations"], output_names=["panorama"], opset_version=20, dynamo=False)
    metadata = {"runtime":"onnxruntime-android","input_images":[1,args.num_frames,3,INPUT_H,INPUT_W],"input_rotations":[1,args.num_frames,3,3],"pose_representation":"camera-to-world rotation matrices derived from yaw_pitch_roll_degrees","input_normalization":{"scale":"1/255","mean":[0.0,0.0,0.0],"std":[1.0,1.0,1.0]},"letterbox":{"width":INPUT_W,"height":INPUT_H,"interpolation":"bilinear","padding":"zero"},"frame_padding":"repeat_last_valid_frame_and_pose","output_panorama":[1,3,cfg["model"]["pano_height"],cfg["model"]["pano_width"]],"output_range":"0..1 RGB","fisheye_model":"180_degree_equidistant"}
    with open(os.path.join(os.path.dirname(args.output), "model_metadata.json"), "w", encoding="utf-8") as handle: json.dump(metadata, handle, indent=2)
    print(f"Exported {args.output}")


if __name__ == "__main__": main()
