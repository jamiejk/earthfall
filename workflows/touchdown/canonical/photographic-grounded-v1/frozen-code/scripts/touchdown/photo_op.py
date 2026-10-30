"""Rank attributed Commons photo references and retain only explicit camera anchors."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def rank_references(manifest: dict, query: str, require_anchors: bool = True) -> dict:
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    photos = manifest["selected"]
    if not photos:
        raise ValueError("No photo-op references available")
    model = SentenceTransformer(os.getenv("TOUCHDOWN_EMBED_MODEL", "google/embeddinggemma-2"),
        config_kwargs={"audio_config": None}, model_kwargs={"torch_dtype": torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32},
        device="cuda" if torch.cuda.is_available() else "cpu")
    inputs = [query]
    for photo in photos:
        if hashlib.sha256(Path(photo["local_path"]).read_bytes()).hexdigest() != photo["sha256"]:
            raise ValueError(f"Reference photo checksum changed: {photo['page_id']}")
        with Image.open(photo["local_path"]) as image:
            inputs.append({"image": image.convert("RGB").copy()})
    vectors = model.encode(inputs, normalize_embeddings=True, show_progress_bar=False)
    ranked = sorted(({**photo, "photo_op_similarity": round(float(vector @ vectors[0]), 5)}
                     for photo, vector in zip(photos, vectors[1:])),
                    key=lambda photo: (-photo["photo_op_similarity"], photo["page_id"]))
    # Coordinates without an explicit camera template remain visual references,
    # never geometric anchors. A camera anchor describes a search neighbourhood,
    # not the exact position of the figure or a verified public standing area.
    anchors = [{**photo["camera_location"], "page_id": photo["page_id"],
                "similarity": photo["photo_op_similarity"]}
               for photo in ranked if photo.get("camera_location")
               and photo["photo_op_similarity"] >= ranked[0]["photo_op_similarity"] - 0.04][:6]
    if not anchors and require_anchors:
        raise ValueError("References found, but none has explicit camera-location metadata")
    return {"query": query, "ranked": ranked, "anchors": anchors,
            "note": "Camera positions are reported by contributors and have unverified accuracy; validate the standing surface and approach separately."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-anchors-ok", action="store_true",
                        help="rank photos for the end still even when none carries camera-location metadata")
    args = parser.parse_args()
    result = rank_references(json.loads(args.manifest.read_text()), args.query, not args.no_anchors_ok)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"anchors": result["anchors"], "photos": len(result["ranked"])}), flush=True)


if __name__ == "__main__":
    main()
