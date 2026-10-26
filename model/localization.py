import numpy as np
from scipy import ndimage


def _patch_defect_density(soft_probs, patch_size=32):
    """
    Downsamples the dequantizer's per-pixel soft class-probability map to a
    patch-level grid (matching the model's own patchify grid) and returns, per
    patch: the fraction of its foreground (die) pixels that are defective, and
    the fraction of the patch that is wafer foreground at all.
    """
    pixel_class = np.argmax(soft_probs, axis=0)
    defect = (pixel_class == 2).astype(np.float32)
    foreground = (pixel_class != 0).astype(np.float32)

    H, W = pixel_class.shape
    gh, gw = H // patch_size, W // patch_size
    density = np.zeros((gh, gw), dtype=np.float32)
    fg_ratio = np.zeros((gh, gw), dtype=np.float32)

    for r in range(gh):
        for c in range(gw):
            d_blk = defect[r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size]
            f_blk = foreground[r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size]
            fg_ratio[r, c] = f_blk.mean()
            density[r, c] = d_blk.sum() / max(f_blk.sum(), 1.0)

    return density, fg_ratio


def find_defect_bboxes(soft_probs, patch_size=32, z_thresh=1.5, min_abs_density=0.55,
                        min_fg_ratio=0.5, max_boxes=3, min_area_patches=2,
                        near_full_density=0.45):
    """
    Derives defect-region bounding boxes from the dequantizer's per-pixel soft
    class-probability map, without any additional detection head.

    Wafer maps in this dataset carry a baseline scatter of defective dies
    everywhere (sensor/process noise), with the labeled pattern showing up as
    a *locally denser* region of defective dies on top of that baseline. So
    rather than thresholding raw per-pixel defect probability (which just
    picks up the baseline noise everywhere), this computes defect density per
    patch (on the same 32x32 grid the transformer itself attends over),
    z-scores each patch's density against the image's own patch-density
    distribution, and boxes the patches that are statistical outliers -
    i.e. "locally unusually defective", which is what a human reading the
    heatmap would call "where the defect is".

    Args:
        soft_probs: (3, H, W) array-like, channel order [background, normal
            die, defective die], as produced by FixedViridisDequantizer.
        patch_size: grid cell size in pixels, should match the model's patch_size.
        z_thresh: minimum z-score (patch density vs. the image's own mean/std)
            for a patch to count as part of a defect region.
        min_abs_density: minimum absolute defect-die fraction a patch must
            also have, so a low-noise image doesn't flag patches purely from
            a tiny std producing inflated z-scores.
        min_fg_ratio: minimum wafer-foreground coverage for a patch to be
            considered at all (excludes patches that are mostly background).
        max_boxes: maximum number of boxes returned (largest regions first).
        min_area_patches: minimum connected-component size, in patches, to
            keep (drops single stray patches).
        near_full_density: if the mean defect density across foreground
            patches exceeds this, the whole wafer is failing (e.g. Near-Full)
            rather than having one small region, so a single box spanning all
            foreground patches is returned instead of z-score outliers.

    Returns:
        List of (x0, y0, x1, y1) pixel boxes, in the same H×W coordinate frame
        as soft_probs, ordered largest-area-first.
    """
    soft_probs = np.asarray(soft_probs)
    density, fg_ratio = _patch_defect_density(soft_probs, patch_size)
    valid = fg_ratio > min_fg_ratio

    if valid.sum() < 3:
        return []

    vals = density[valid]
    mean, std = float(vals.mean()), float(vals.std())

    if mean > near_full_density:
        ys, xs = np.where(valid)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        return [(int(x0 * patch_size), int(y0 * patch_size),
                  int((x1 + 1) * patch_size - 1), int((y1 + 1) * patch_size - 1))]

    if std < 1e-6:
        return []

    z = (density - mean) / std
    hot = valid & (z > z_thresh) & (density > min_abs_density)
    hot = ndimage.binary_closing(hot, structure=np.ones((2, 2)))

    labeled, num_features = ndimage.label(hot)
    boxes = []
    for label_id in range(1, num_features + 1):
        ys, xs = np.where(labeled == label_id)
        if ys.size < min_area_patches:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        boxes.append((x0, y0, x1, y1, ys.size))

    boxes.sort(key=lambda b: b[4], reverse=True)
    boxes = boxes[:max_boxes]
    return [
        (int(x0 * patch_size), int(y0 * patch_size),
         int((x1 + 1) * patch_size - 1), int((y1 + 1) * patch_size - 1))
        for (x0, y0, x1, y1, _area) in boxes
    ]
