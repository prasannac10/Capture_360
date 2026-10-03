import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from panorama.pano_ai.data.ptgui_calibration import PTGuiCalibration, inverse_radius


class PTGuiCalibrationTests(unittest.TestCase):
    def project(self):
        lens = dict(lens=dict(params=dict(projection='rectilinear',focallength=10.,sensordiagonal=20.,
                    a=0.,b=0.,c=0.,mirrored=False,croprectanglesize=None)),
                    shift=dict(params=dict(longside=0.,shortside=0.)),shear=dict(params=dict(hshear=0.,vshear=0.)))
        groups = [dict(size=[64,48], globallens=0, images=[dict(filename=f'{i}.DNG')],
                       position=dict(params=dict(yaw=0.,pitch=0.,roll=0.))) for i in range(2)]
        points = [dict(t=0, **{'0':[0,0,x,y],'1':[1,0,x,y]}) for y in (10,20,30,40) for x in (8,16,24,32,40)]
        return dict(project=dict(panoramaparams=dict(projection='equirectangular',hfov=360,vfov=180),
                                 imagegroups=groups,globallenses=[lens],controlpoints=points))

    def test_rectification_identity_and_bad_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'Panorama.pts';project=self.project();path.write_text(json.dumps(project))
            calibration=PTGuiCalibration(path)
            rgb=np.arange(64*48*3,dtype=np.uint8).reshape(48,64,3)
            actual,frame,zoom=calibration.rectify('0',rgb)
            np.testing.assert_array_equal(actual,rgb)
            self.assertEqual(zoom,1.)
            self.assertEqual(frame.intrinsics[:2],[40.,40.])
            project['project']['imagegroups'][1]['position']['params']['yaw']=30
            path.write_text(json.dumps(project))
            with self.assertRaisesRegex(ValueError,'verification failed'):
                PTGuiCalibration(path)

    def test_radial_inverse_and_unsupported_lens(self):
        radius=np.linspace(0,1.5,50);a,b,c=.01,-.03,.04
        distorted=(a*radius**3+b*radius**2+c*radius+1-a-b-c)*radius
        np.testing.assert_allclose(inverse_radius(distorted,(a,b,c)),radius,atol=1e-10)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'Panorama.pts';project=self.project()
            project['project']['globallenses'][0]['shear']['params']['hshear']=.1
            path.write_text(json.dumps(project))
            with self.assertRaisesRegex(ValueError,'shear'):
                PTGuiCalibration(path)
