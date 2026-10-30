"""Does the end still's background read as a photograph or as a 3D render? (Jamie 18 Oct: "the environs have a 3D-model
look"; the buttes kept the terrain model's flat-shaded facets.)

EmbeddingGemma (already used for the view and photo ranking) compares image crops with two descriptions, a real
photograph of the landscape and a low-poly 3D render. A whole frame does not work: the photographic figure dominates
the score (the faceted v12 still scored above the photographic v14). The band of distant landforms either side of the
figure does: margins (photo minus render) on Monument Valley were v12 faceted -0.03/-0.02, the raw terrain render
-0.04/-0.02, v13 smooth cones +0.03/+0.03, v14 photographic +0.06/+0.08.

Run in the embedding environment (TOUCHDOWN_EMBED_DEPS): python -m touchdown.realism --images a.png b.png --output r.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

W, H = 960, 544
PHOTO = "A real photograph of the distant landscape: real rock, ground and buildings with natural texture, light and haze."
RENDER = ("A 3D computer render of a low-poly terrain model: flat-shaded triangular facets, smooth CGI surfaces, "
          "video game graphics.")


def background_tiles(image: np.ndarray, box: tuple | None, band: tuple[float, float] = (0.2, 0.62),
                     gap: int = 20) -> list[np.ndarray]:
    """The band of distant landforms (rows ``band`` of the frame), split left and right of the figure's columns
    (``box`` = x0, y0, x1, y1 at W x H, or the middle 12% when no figure is found). BGR in, BGR tiles out; a tile
    narrower than a fifth of the frame is dropped."""
    im = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
    strip = im[int(H * band[0]):int(H * band[1])]
    x0, x1 = (box[0] - gap, box[2] + gap) if box else (int(W * 0.44), int(W * 0.56))
    tiles = [strip[:, :max(x0, 0)], strip[:, min(x1, W):]]
    return [t for t in tiles if t.shape[1] >= W // 5]


def verdict(margins: list[float], minimum: float) -> tuple[bool, str]:
    """Passes when every background tile leans photograph by at least ``minimum``."""
    if not margins:
        return True, "no background tile to judge"
    worst = min(margins)
    if worst < minimum:
        return False, f"background reads as a 3D render (photo-minus-render margin {worst:+.3f} < {minimum:+.3f})"
    return True, f"background reads as a photograph (worst margin {worst:+.3f})"


def score(paths: list[Path]) -> list[dict]:
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    from .end_register import person_box
    model = SentenceTransformer(os.getenv("TOUCHDOWN_EMBED_MODEL", "google/embeddinggemma-2"),
                                config_kwargs={"audio_config": None},
                                model_kwargs={"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32},
                                device="cuda" if torch.cuda.is_available() else "cpu")
    text = model.encode([PHOTO, RENDER], normalize_embeddings=True)
    out = []
    for p in paths:
        im = cv2.imread(str(p))
        small = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        tiles = background_tiles(im, person_box(small))
        margins = []
        for t in tiles:
            v = model.encode([{"image": Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))}], normalize_embeddings=True)[0]
            s = v @ text.T
            margins.append(round(float(s[0] - s[1]), 4))
        out.append({"image": str(p), "margins": margins})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(score(a.images), indent=2) + "\n")


if __name__ == "__main__":
    main()
