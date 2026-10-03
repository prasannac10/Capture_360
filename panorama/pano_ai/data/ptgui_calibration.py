"""Restricted PTGui JSON rectilinear importer, verified against control points.

Shift convention: https://www.ptgui.com/parsepts.html
Radial model: https://panotools.org/dersch/barrel/barrel.html
"""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from panorama.stitching.capture import CameraFrame, CV_BASIS
from ..models.tile_spherical import ypr_to_rot


def inverse_radius(radius, coefficients):
    a, b, c = coefficients
    d = 1 - a - b - c
    result = np.array(radius, dtype=np.float64, copy=True)
    for _ in range(12):
        derivative = 4*a*result**3 + 3*b*result**2 + 2*c*result + d
        if np.any(derivative <= 0):
            raise ValueError('Non-monotonic PTGui radial model')
        result -= (a*result**4+b*result**3+c*result**2+d*result-radius)/derivative
    return result


class PTGuiCalibration:
    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        project = data['project']
        panorama = project['panoramaparams']
        if (panorama['projection'] != 'equirectangular' or panorama['hfov'] != 360
                or panorama['vfov'] != 180 or panorama.get('outputcrop', [0,0,1,1]) != [0,0,1,1]):
            raise ValueError('Require uncropped full-sphere equirectangular PTGui project')
        self.records = {}
        groups = project['imagegroups']
        for index, group in enumerate(groups):
            if len(group['images']) != 1:
                raise ValueError('Bracketed PTGui groups are not supported')
            image = group['images'][0]
            if not image.get('include', True) or image.get('ispatch', False) or group.get('maskbitmap'):
                raise ValueError('Excluded images, patches and PTGui masks require separate handling')
            if any(group.get(k) is not None for k in ('individuallens','individualshift','individualshear','individualcropcenter')):
                raise ValueError('Per-image lens overrides are not supported')
            lens = project['globallenses'][group['globallens']]
            params = lens['lens']['params']
            if params['projection'] != 'rectilinear' or params.get('mirrored') or params.get('croprectanglesize'):
                raise ValueError('Require unmirrored uncropped rectilinear sources')
            if any(params.get('cropcenteroffset', {}).values()):
                raise ValueError('PTGui crop offsets require separate handling')
            if any(lens['shear']['params'].values()):
                raise ValueError('PTGui shear is not supported')
            position = group['position']['params']
            if any(position.get(k, 0) != 0 for k in ('vpx','vpy','vpd','vppan','vptilt')):
                raise ValueError('PTGui viewpoint correction is not supported')
            w, h = group['size']
            diagonal = np.hypot(w, h)
            shift = lens['shift']['params']
            sx, sy = shift['longside'], shift['shortside']
            if h > w:
                sx, sy = ((sy, -sx) if project['portraitcameraorientation'] == 'counterclockwise' else (-sy, sx))
            rotation = ypr_to_rot(torch.tensor([[position['yaw'], position['pitch'], position['roll']]], dtype=torch.float64))[0].numpy()
            name = Path(image['filename'].replace('\\','/')).stem
            if name.lower() in self.records:
                raise ValueError('Duplicate PTGui image stems')
            self.records[name.lower()] = dict(index=index, size=(w,h), f=params['focallength']/params['sensordiagonal']*diagonal,
                center=np.array([(w-1)/2+sx*diagonal, (h-1)/2+sy*diagonal]),
                coefficients=np.array([params['a'],params['b'],params['c']]), rotation=rotation, name=name)
        # Control points check geometry before any RAW files are decoded.
        by_index = {r['index']: r for r in self.records.values()}
        errors = []
        for cp in project['controlpoints']:
            if cp['t'] != 0:
                continue
            rays = []
            for key in ('0','1'):
                index, bracket, u, v = cp[key]
                if bracket != 0:
                    raise ValueError('Bracketed control points are not supported')
                r = by_index[index]
                xy = np.array([u,v])-r['center']
                radius = np.linalg.norm(xy)/(min(r['size'])/2)
                ideal = inverse_radius(radius, r['coefficients'])
                xy *= ideal/radius if radius else 1
                ray = r['rotation'] @ np.r_[xy/r['f'], 1.]
                rays.append(ray/np.linalg.norm(ray))
            errors.append(np.degrees(np.arccos(np.clip(np.dot(*rays), -1, 1))))
        if len(errors) < 20:
            raise ValueError('Require at least 20 normal control points for import verification')
        self.validation = dict(control_points=len(errors), median_angular_error_degrees=float(np.median(errors)),
                               p90_angular_error_degrees=float(np.percentile(errors,90)))
        if self.validation['median_angular_error_degrees'] > .15 or self.validation['p90_angular_error_degrees'] > .5:
            raise ValueError(f'PTGui calibration verification failed: {self.validation}')

    def rectify(self, name, rgb):
        r = self.records[name.lower()]
        w, h = r['size']
        if rgb.shape[:2] != (h,w):
            raise ValueError('Full-size RAW decoding must match PTGui image dimensions/orientation')
        # Keep all output pixels valid: modest focal zoom avoids synthetic borders.
        yy, xx = np.indices((h,w), dtype=np.float32)
        cx, cy = (w-1)/2, (h-1)/2
        a,b,c = r['coefficients']; d=1-a-b-c
        for zoom in np.arange(1., 1.201, .005):
            dx, dy = (xx-cx)/zoom, (yy-cy)/zoom
            radius = np.hypot(dx,dy)/(min(w,h)/2)
            factor = ((a*radius+b)*radius+c)*radius+d
            mx, my = dx*factor+r['center'][0], dy*factor+r['center'][1]
            if mx.min() >= 0 and my.min() >= 0 and mx.max() <= w-1 and my.max() <= h-1:
                break
        else:
            raise ValueError('Lens correction needs more than 20 percent zoom; refusing silent crop')
        corrected = cv2.remap(rgb, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LANCZOS4)
        frame = CameraFrame(r['name'], '', w,h, [r['f']*zoom,r['f']*zoom,cx,cy],
                            (CV_BASIS@r['rotation']@CV_BASIS).tolist(), [0.,0.,0.], translation_known=False)
        return corrected, frame.validate(), float(zoom)
