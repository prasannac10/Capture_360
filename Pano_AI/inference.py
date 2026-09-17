"""CLI compatibility entry point for the authoritative high-resolution inference path."""
import argparse
from reference_inference import run_reference_inference

if __name__ == '__main__':
    p=argparse.ArgumentParser(description='Capture360 high-resolution inference')
    p.add_argument('--session',required=True); p.add_argument('--config',default='config.yaml'); p.add_argument('--output',required=True); p.add_argument('--device',choices=['cpu','cuda'],default=None); p.add_argument('--seed',type=int,default=0); p.add_argument('--no-corrections',action='store_true'); a=p.parse_args()
    run_reference_inference(a.session,a.config,a.output,a.device,a.seed,not a.no_corrections)
