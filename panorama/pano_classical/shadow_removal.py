"""Explicitly masked shadow relighting; never classify darkness as a defect."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def remove_shadows(image, mask, max_gain=3.0):
    """Estimate illumination from nearby unmasked pixels and retain RGB texture.

    Mask is float [0,1]. Zero pixels are preserved exactly. This approximates
    illumination on similar surfaces, not object removal or missing RGB recovery.
    """
    image, mask = np.asarray(image), np.asarray(mask, dtype=np.float32)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected uint8 BGR image')
    if mask.shape != image.shape[:2] or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError('Shadow mask must match image and contain finite values in [0,1]')
    if not np.isfinite(max_gain) or not 1 <= max_gain <= 8:
        raise ValueError('max_gain must be between 1 and 8')
    if not mask.any():
        return image.copy()
    if np.mean(mask > 0) > .5:
        raise ValueError('Shadow mask must leave more than half the panorama as reference')
    h, w = mask.shape
    scale = min(1., 1024 / w)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    low = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    # AREA retains thin marked regions instead of losing them through nearest sampling.
    selected = (cv2.resize(mask, size, interpolation=cv2.INTER_AREA) > 0).astype(np.uint8) * 255
    selected = cv2.dilate(selected, np.ones((3, 3), np.uint8))
    if np.all(selected):
        raise ValueError('Mask leaves no illumination reference at working resolution')
    pad = min(128, size[0])
    extended = cv2.copyMakeBorder(low, 0, 0, pad, pad, cv2.BORDER_WRAP)
    extended_mask = cv2.copyMakeBorder(selected, 0, 0, pad, pad, cv2.BORDER_WRAP)
    light = cv2.inpaint(extended, extended_mask, 5, cv2.INPAINT_NS)[:, pad:-pad]
    light = cv2.GaussianBlur(light, (0, 0), 2)
    # Smooth the observed lighting separately; gain changes illumination, not texture.
    observed = cv2.GaussianBlur(extended, (0, 0), 2)[:, pad:-pad]
    gain = np.clip(light / np.maximum(observed, 1), 1, max_gain)
    gain = cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR)
    result = image.copy()
    # Work one channel at a time to keep native-resolution memory bounded.
    for channel in range(3):
        corrected = np.clip(image[..., channel].astype(np.float32) * gain, 0, 255)
        blended = np.rint(image[..., channel] * (1 - mask) + corrected * mask).astype(np.uint8)
        result[..., channel][mask > 0] = blended[mask > 0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--mask', required=True, help='Exact-size grayscale mask: white=shadow, black=protected')
    parser.add_argument('--out', required=True, help='New lossless PNG output')
    parser.add_argument('--max-gain', type=float, default=3.)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists() or output.suffix.lower() != '.png':
        raise ValueError('Use a new PNG output path to preserve originals and protected pixels')
    image = cv2.imread(args.image, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError('Image and mask must be readable')
    result = remove_shadows(image, mask.astype(np.float32) / 255, args.max_gain)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), result):
        raise OSError(f'Could not write {output}')
    print(output.resolve())


if __name__ == '__main__':
    main()
