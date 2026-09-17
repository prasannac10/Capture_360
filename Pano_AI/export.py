"""Deployment export notes for the variable-resolution pipeline.

The native tiled reference path is not exported as the old 224x224 dynamic-N
ONNX graph. Exporting it requires a deployment runtime that supports the
re-iterable tile stream and spherical projection. Keep Python as the reference
implementation until that contract is frozen.
"""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='artifacts/tiled_runtime_contract.json'); a=p.parse_args(); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'input':'native-resolution frames','tiling':'1024x1024 with 128 overlap','frame_range':[4,30],'projections':['fisheye_180','pinhole'],'output':[12000,6000],'runtime_status':'python-reference','onnx_status':'not_exported_for_native_tiled_path'},indent=2),encoding='utf-8'); print(out)
if __name__=='__main__': main()
