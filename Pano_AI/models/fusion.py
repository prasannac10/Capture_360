from models.spherical import spherical_project


class SphericalFusion:
    """Canonical camera-aware spherical feature projection wrapper."""

    def __call__(self, feats, rotations, pano_h, pano_w, frame_mask=None, camera_params=None):
        return spherical_project(feats, rotations, pano_h, pano_w, frame_mask, camera_params)
