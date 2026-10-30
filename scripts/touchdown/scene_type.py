"""Classify the requested shot as a landmark, urban or nature scene with EmbeddingGemma 2 (Jamie 17 Oct).

The user picks the spot and describes it; the class decides which ground features the photo-grounded stills are
told to preserve (buildings and streets, a monument's shape, or landforms and vegetation). Text only: the request
(place name, catalogue kind, scene words) is compared with a few prototype descriptions per class. Run in a separate
process so the model releases VRAM before FLUX and LTX, like view_rank.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROTOTYPES = {
    "landmark": ["a famous landmark, monument or statue that visitors come to photograph",
                 "standing in front of a well-known monument, tower, bridge or historic building",
                 "a world-famous sight seen up close"],
    "urban": ["a city street, square or park surrounded by buildings",
              "an urban scene with buildings, roads, pavements and street furniture",
              "downtown among office towers and shops"],
    "nature": ["a natural landscape of desert, mountains, forest, canyon or coast",
               "open wild terrain with rocks, cliffs and vegetation",
               "a remote viewpoint over a valley or wilderness"],
}


def classify(text: str, model_id: str = "google/embeddinggemma-2") -> dict:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, config_kwargs={"audio_config": None},
                                model_kwargs={"torch_dtype": torch.bfloat16 if torch.cuda.is_available()
                                              and torch.cuda.is_bf16_supported() else torch.float32},
                                device="cuda" if torch.cuda.is_available() else "cpu")
    labels = list(PROTOTYPES)
    phrases = [p for label in labels for p in PROTOTYPES[label]]
    emb = model.encode([text] + phrases, normalize_embeddings=True, show_progress_bar=False)
    sims = np.asarray(emb[1:] @ emb[0], dtype=np.float32)
    scores, k = {}, 0
    for label in labels:
        n = len(PROTOTYPES[label])
        scores[label] = round(float(sims[k:k + n].mean()), 4)
        k += n
    best = max(scores, key=scores.get)
    ordered = sorted(scores.values(), reverse=True)
    return {"type": best, "scores": scores, "margin": round(ordered[0] - ordered[1], 4), "text": text,
            "model": model_id}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = classify(a.text, os.getenv("TOUCHDOWN_EMBED_MODEL", "google/embeddinggemma-2"))
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("type", "scores", "margin")}), flush=True)


if __name__ == "__main__":
    main()
