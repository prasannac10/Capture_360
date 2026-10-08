"""Compare native ownership, photometric blending and saved neural output crops."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from ..data.tile_dataset import VariableTilePanoramaDataset
from ..data.native_rgb import NativeRGBSource
from .metrics import image_quality, seam_quality


def diagnose(scene, config_path, output, region, prediction=None):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    model = config['model']
    size = (model['output_height'], model['output_width'])
    top, left, h, w = region
    if min(top,left) < 0 or min(h,w) <= 0 or top+h > size[0] or left+w > size[1]:
        raise ValueError('Region must lie within the configured panorama')
    dataset = VariableTilePanoramaDataset(scene.parent, has_gt=True, input_config=config['input'])
    sample = dataset[dataset.scenes.index(scene)]
    if tuple(sample['gt_panorama'].shape[-2:]) != size:
        raise ValueError('Reference must match configured native output dimensions')
    if output.exists():
        raise FileExistsError('Use a new diagnostic output directory')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = sample['gt_panorama'][None, :, top:top+h, left:left+w].to(device)
    images = {}
    for name, blend in [('ownership',False), ('photometric',True)]:
        sampler = NativeRGBSource(sample, photometric_blending=blend)
        images[name], coverage, _ = sampler(region,size,device)
        seam = sampler.last_diagnostics['seam_mask']
    if prediction:
        with Image.open(prediction) as image:
            if image.size != (size[1],size[0]):
                raise ValueError('Saved prediction must match native output size; do not resize it')
            crop = np.array(image.convert('RGB').crop((left,top,left+w,top+h)),copy=True)
        images['trained'] = torch.from_numpy(crop).permute(2,0,1)[None].to(device).float()/255
    output.mkdir(parents=True)
    report = dict(scene=str(scene), region=list(region), output_size=list(size),
                  device=str(device), prediction=str(prediction) if prediction else None, metrics={})
    for name, rgb in dict(images, reference=target).items():
        pixels = (rgb[0].permute(1,2,0).cpu().numpy().clip(0,1)*255).round().astype(np.uint8)
        Image.fromarray(pixels).save(output/(name+'.png'))
        if name != 'reference':
            report['metrics'][name] = dict(image_quality(rgb,target,coverage), **seam_quality(rgb,target,seam))
    for name, mask in [('coverage',coverage), ('seam',seam)]:
        Image.fromarray((mask[0,0].cpu().numpy()*255).astype(np.uint8)).save(output/(name+'.png'))
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--region',type=int,nargs=4,required=True,metavar=('TOP','LEFT','HEIGHT','WIDTH'))
    p.add_argument('--prediction',help='Saved initial_panorama.png from neural inference, before correction')
    args=p.parse_args()
    print(json.dumps(diagnose(args.scene,args.config,args.out,tuple(args.region),args.prediction),indent=2))


if __name__ == '__main__':
    main()
