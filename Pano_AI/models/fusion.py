from models.spherical import spherical_project


class SphericalFusion:
    """Compatibility wrapper; spherical_project has one canonical implementation."""
    def __call__(self, feats, rotations, pano_h, pano_w, frame_mask=None):
        return spherical_project(feats, rotations, pano_h, pano_w, frame_mask)
