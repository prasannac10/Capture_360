"""Classical OpenCV baseline for high-resolution Capture360 stitching."""
from pathlib import Path
import cv2
import numpy as np
from utils.high_resolution import OUTPUT_SIZE

def fisheye_to_equirectangular(image, output_size=OUTPUT_SIZE, fov_degrees=180.0):
    w,h=output_size; yy,xx=np.meshgrid(np.arange(h,dtype=np.float32),np.arange(w,dtype=np.float32),indexing='ij'); lon=(xx/w-.5)*2*np.pi; lat=(.5-yy/h)*np.pi; x=np.cos(lat)*np.sin(lon); y=np.sin(lat); z=np.cos(lat)*np.cos(lon); theta=np.arctan2(x,z); phi=np.arctan2(y,np.sqrt(x*x+z*z)); scale=np.sqrt(theta*theta+phi*phi)/(np.pi/2); radius=scale*(min(image.shape[:2])/2); sx=image.shape[1]/2+radius*np.sin(theta)/np.maximum(scale,1e-6); sy=image.shape[0]/2-radius*np.sin(phi)/np.maximum(scale,1e-6); valid=scale<=1; sx=np.mod(sx,image.shape[1]).astype(np.float32); sy=np.clip(sy,0,image.shape[0]-1).astype(np.float32); out=cv2.remap(image,sx,sy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT); out[~valid]=0; return out

def stitch_fisheye_files(image_paths,output_path,output_size=OUTPUT_SIZE,fov_degrees=180.0):
    frames=[cv2.imread(str(p),cv2.IMREAD_COLOR) for p in image_paths]
    if not frames or any(x is None for x in frames): raise ValueError('Could not read one or more fisheye images')
    acc=np.zeros((output_size[1],output_size[0],3),np.float32); weight=np.zeros(output_size[::-1],np.float32)
    for frame in frames:
        projected=fisheye_to_equirectangular(frame,output_size,fov_degrees); mask=np.any(projected!=0,axis=2).astype(np.float32); acc+=projected.astype(np.float32)*mask[...,None]; weight+=mask
    pano=(acc/np.maximum(weight[...,None],1)).clip(0,255).astype(np.uint8); Path(output_path).parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(output_path),pano); return pano

def stitch_pinhole_files(image_paths,output_size=OUTPUT_SIZE):
    frames=[cv2.imread(str(p),cv2.IMREAD_COLOR) for p in image_paths]
    if not frames or any(x is None for x in frames): raise ValueError('Could not read one or more input images')
    stitcher=cv2.Stitcher_create(cv2.Stitcher_PANORAMA); status,pano=stitcher.stitch(frames)
    if status != cv2.Stitcher_OK: raise RuntimeError(f'OpenCV Stitcher failed with status {status}')
    return cv2.resize(pano,output_size,interpolation=cv2.INTER_LANCZOS4)
