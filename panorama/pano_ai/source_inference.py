"""Source-preserving AI route with checked pairwise proposals and RGB fallback."""
import json
from pathlib import Path
import cv2
import torch
from PIL import Image

from panorama.stitching.defects import load_defect_masks, union_masks, save_defect_masks
from panorama.pano_classical.pose_stitcher import stitch_pose_files
from .data.tile_dataset import VariableTilePanoramaDataset
from .view_refinement import GuardedViewRefiner
from .highres_correction import HighResolutionCorrectionPipeline


def run_source_inference(scene, config, config_path, output_dir=None, profile='auto'):
    scene, config_path = Path(scene).resolve(), Path(config_path).resolve()
    model_cfg = config['model']
    dataset = VariableTilePanoramaDataset(scene.parent, False, model_cfg['tile_size'], model_cfg['tile_overlap'],
                                          input_config=config['input'], profile=profile,
                                          exposure_config={'enabled': False})
    sample = dataset[next(i for i, p in enumerate(dataset.scenes) if p.resolve() == scene)]
    output = Path(output_dir or config_path.parent / config['inference']['output_dir']) / scene.name
    output.mkdir(parents=True, exist_ok=True)
    capture = sample['capture']
    capture.save(output / 'capture.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    settings = config.get('ai_pipeline', {})
    views = GuardedViewRefiner(settings.get('learned_views', {}), config_path, device)
    corrections = HighResolutionCorrectionPipeline(config, config_path, device)
    options = dict(config.get('classical_pose', {}))
    options.update(settings.get('geometry', {}))
    options['local_alignment'] = False  # The guarded learned proposal is the AI alignment path.
    # Preserve the original capture manifest for guided heading recovery.
    poses_path = capture.root / 'ar_poses.jsonl'
    if not poses_path.exists():
        poses_path = output / 'capture.json'
    raw = stitch_pose_files(sample['frame_paths'], output / 'initial_panorama.png',
                             (model_cfg['output_width'], model_cfg['output_height']), poses_path, options,
                             view_refiner=views if views.models else None)
    coverage = cv2.imread(str(output / 'initial_panorama_coverage.png'), cv2.IMREAD_GRAYSCALE)
    masks = load_defect_masks(scene, raw.shape[:2], coverage)
    image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    Image.fromarray(image).save(output / 'initial_panorama.tiff', compression='tiff_deflate')
    repaired, report = corrections.run(image, union_masks(masks, ('missing_coverage', 'photographer', 'other')),
                                       model_cfg['tile_size'], model_cfg['tile_overlap'],
                                       auxiliary={'ghost_mask': masks.get('moving_objects')}, output_dir=output,
                                       defect_masks=masks)
    Image.fromarray(repaired).save(output / 'final_panorama.png')
    Image.fromarray(repaired).save(output / 'final_panorama.tiff', compression='tiff_deflate')
    geometry = json.loads((output / 'initial_panorama_geometry.json').read_text())
    metadata = {'engine': 'ai', 'mode': 'source_preserving', 'camera_contract': 'capture.json',
                'stitching': geometry, 'corrections': report,
                'defects': save_defect_masks(output, masks),
                'learned_tasks': sorted(views.models),
                'checkpoints': views.checkpoints,
                'fallback': 'geometric_projection_and_seams', 'translation_used': False,
                'depth_used': False}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(output / 'final_panorama.png')
    return output
