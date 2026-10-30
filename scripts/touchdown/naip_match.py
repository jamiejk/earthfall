"""Colour-match the nested NAIP squares before the Blender terrain render (17 Oct).

The far terrain mesh takes, per polygon, the NAIP square that covers it (900, 2700, 8100, 24300, 72900 m). Squares
of different sizes come from different mosaics and capture dates, so where the mesh switches square there is a hard
line, and the coarser square can be redder (FLUX 3 v3 Monument Valley, dive f55-75). Each coarser square is matched
to the finer square it contains, finest outward: per Lab channel, the coarse square's central overlap is given the
mean and spread of the (downsampled) finer square, and the same affine change is applied to the whole coarse square.
Mosaic seams inside one square (capture dates meeting in straight lines) are flattened on the squares from
``flatten_from`` up: NAIP tiles meet in straight north-south/east-west lines, so ``flatten`` of each row's and each
column's deviation from a smoothed median colour profile is removed; 2D detail and large-scale colour stay. Copies go to ``<imagery>/matched/``; the originals are not touched.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def match(imagery: Path, levels=(900, 2700, 8100, 24300, 72900), out: str = "matched", flatten_from: int = 8100,
          flatten: float = 0.9) -> Path:
    folder = imagery / out
    folder.mkdir(exist_ok=True)
    finer = None  # Lab float of the previous (finer, already matched) square
    for width in levels:
        src = cv2.imread(str(imagery / f"naip_{width:05d}.png"))
        if src is None:
            raise FileNotFoundError(imagery / f"naip_{width:05d}.png")
        lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
        if finer is not None:
            n = lab.shape[0]
            ratio = width / prev_width
            inner = int(round(n / ratio))
            a = (n - inner) // 2
            centre = lab[a:a + inner, a:a + inner]
            small = cv2.resize(finer, (inner, inner), interpolation=cv2.INTER_AREA)
            valid = (small.sum(axis=2) > 1) & (centre.sum(axis=2) > 1)  # skip no-data borders
            if valid.mean() > 0.2:
                for c in range(3):
                    cm, cs = centre[..., c][valid].mean(), centre[..., c][valid].std() + 1e-3
                    fm, fs = small[..., c][valid].mean(), small[..., c][valid].std() + 1e-3
                    gain = float(np.clip(fs / cs, 0.6, 1.6))
                    lab[..., c] = (lab[..., c] - cm) * gain + fm
                lab = np.clip(lab, 0, 255)
        if width >= flatten_from:
            # NAIP capture tiles meet in straight north-south / east-west lines, so a mosaic seam inside a square
            # is a band of rows or columns with a different colour (a redder strip across the 8100 and 24300 m
            # squares at Monument Valley: a hard line in the high dive). Take each row's (then each column's)
            # median colour, compare it with a heavily smoothed profile, and remove ``flatten`` of the difference:
            # straight bands go, 2D detail and the large-scale colour stay. (Pulling the whole square to its mean
            # turned the far buttes olive and the FLUX 3 end still lost them: v4 Monument Valley, shift 90 px.)
            n = lab.shape[0]
            for axis in (1, 0):  # rows (median across columns), then columns
                profile = np.median(lab, axis=axis)  # (n, 3)
                smooth = np.median(profile, axis=0)[None, :]  # a band is a quarter of the square: compare with the whole
                fine = cv2.GaussianBlur(profile.reshape(n, 1, 3), (0, 0), sigmaX=0.1, sigmaY=1.5).reshape(n, 3)
                dev = fine - smooth
                lab = lab - flatten * (dev[:, None, :] if axis == 1 else dev[None, :, :])
            lab = np.clip(lab, 0, 255)
        cv2.imwrite(str(folder / f"naip_{width:05d}.png"),
                    cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR))
        finer, prev_width = lab, width
    return folder
