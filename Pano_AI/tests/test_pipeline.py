"""Smoke test the complete modular pipeline on local sample sets.
Usage: python -m tests.test_pipeline /path/to/data --limit 3
"""
import argparse
from pathlib import Path
import cv2
from pipeline import CorrectionPipeline
from stitching.opencv_stitcher import stitch_fisheye_files

def main():
    p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--limit',type=int,default=3); a=p.parse_args()
    scenes=sorted(x for x in Path(a.root).iterdir() if (x/'images').is_dir())[:a.limit]
    if not scenes: raise RuntimeError('No sample scenes found')
    pipe=CorrectionPipeline({'glare':False,'dots':True,'nadir_zenith':False,'color':False,'sharpen':True})
    for scene in scenes:
        frames=sorted((scene/'images').glob('*')); baseline=stitch_fisheye_files(frames,scene/'_smoke_baseline.png'); out=pipe.run(baseline); assert out.ndim==3 and out.shape[2]==3; cv2.imwrite(str(scene/'_smoke_output.png'),cv2.cvtColor(out,cv2.COLOR_RGB2BGR))
    print(f'PIPELINE SMOKE PASS: {len(scenes)} scenes')

if __name__=='__main__': main()
