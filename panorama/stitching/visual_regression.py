"""Reference-based edge/texture/seam checks and exact protected-pixel checks."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def metrics(candidate, reference, original=None, mask=None):
    if candidate.shape != reference.shape or candidate.ndim != 3:
        raise ValueError('Candidate and reference must have identical RGB dimensions')
    a, b = candidate.astype(np.float32) / 255, reference.astype(np.float32) / 255
    def dx(x):
        return np.roll(x, -1, axis=1) - x
    def dy(x):
        return x[1:] - x[:-1]
    def highpass(x):
        padded = cv2.copyMakeBorder(x, 0, 0, 4, 4, cv2.BORDER_WRAP)
        return x - cv2.GaussianBlur(padded, (9, 9), 1.2)[:, 4:-4]
    result = dict(rgb_mae=float(np.abs(a - b).mean()),
                  edge_error=float((np.abs(dx(a) - dx(b)).mean() + np.abs(dy(a) - dy(b)).mean()) / 2),
                  texture_error=float(np.abs(highpass(a) - highpass(b)).mean()),
                  longitude_join_error=float(np.abs((a[:, 0] - a[:, -1]) - (b[:, 0] - b[:, -1])).mean()))
    if (original is None) != (mask is None):
        raise ValueError('Unmasked preservation requires both original and defect mask')
    if original is not None:
        if original.shape != candidate.shape or mask.shape != candidate.shape[:2]:
            raise ValueError('Original and mask must use identical output coordinates')
        if not np.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
            raise ValueError('Defect mask must be finite [0,1]')
        difference = np.abs(candidate.astype(np.float64) - original.astype(np.float64))
        protected = mask == 0
        result['unmasked_changed_pixels'] = int(np.count_nonzero(np.any(difference > 0, axis=-1) & protected))
        result['unmasked_max_error_8bit'] = float(difference[protected].max()) if protected.any() else 0.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--reference', required=True, help='Aligned ground truth or an explicitly chosen baseline')
    parser.add_argument('--original', help='Input to the masked restoration stage (lossless PNG)')
    parser.add_argument('--mask', help='Exact stage mask; zero pixels are protected')
    parser.add_argument('--out', required=True)
    parser.add_argument('--limits', help='JSON maximum metric values; exit nonzero on regressions')
    args = parser.parse_args()
    def read(path, gray=False):
        a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if a is None:
            raise ValueError(f'Cannot read {path}')
        return a
    candidate, reference = read(args.candidate), read(args.reference)
    result = metrics(candidate, reference, read(args.original) if args.original else None,
                     read(args.mask, True).astype(np.float32) / 255 if args.mask else None)
    limits = json.loads(Path(args.limits).read_text()) if args.limits else {}
    if set(limits) - set(result):
        raise ValueError('Limits name metrics not evaluated by this run')
    if any(not np.isfinite(v) or v < 0 for v in limits.values()):
        raise ValueError('Limits must be finite and nonnegative')
    failures = {k: result[k] for k, maximum in limits.items() if result[k] > maximum}
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(dict(metrics=result, failures=failures,
            reference=str(Path(args.reference).resolve()), note='Differences from a baseline are not proof of quality improvement.'), indent=2))
    # Native-resolution crops preserve the defects that thumbnail resizing hides.
    h, w = candidate.shape[:2]
    ch, cw = min(512, h), min(768, w)
    regions = {'centre': ((h-ch)//2, (w-cw)//2), 'bottom': (h-ch, (w-cw)//2), 'longitude_join': ((h-ch)//2, w-cw//2)}
    for name, (y, x) in regions.items():
        columns = np.arange(x, x+cw) % w
        pair = np.concatenate([reference[y:y+ch, columns], candidate[y:y+ch, columns]], axis=1)
        if not cv2.imwrite(str(output / f'{name}_reference_candidate.png'), pair):
            raise OSError('Could not save visual regression crop')
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
