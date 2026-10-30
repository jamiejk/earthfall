"""Rank measured final-view previews against a scene description with EmbeddingGemma 2.

Run in a separate process before FLUX/LTX so the embedding model releases VRAM.
The geometry shortlist is computed by terrain_route; this module only orders it.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def select_view(plan: list[dict], previews: Path, query: str, *, contacts: Path | None = None,
                route_query: str | None = None,
                avoid_pavement: bool = False,
                model_id: str = "google/embeddinggemma-2", max_geometry_gap: float = 1.5) -> dict:
    import numpy as np
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    if not plan:
        raise ValueError("No measured candidate views to rank")
    best_geometry = min(item["score"] for item in plan)
    eligible = [i for i, item in enumerate(plan) if item["score"] <= best_geometry + max_geometry_gap]
    rejected = []
    if avoid_pavement:
        import cv2
        for index in eligible.copy():
            preview = cv2.imread(str(previews / f"{index + 1:04d}.png"))
            if preview is None:
                raise ValueError("Missing final-view preview")
            hsv = cv2.cvtColor(cv2.resize(preview, (960, 544)), cv2.COLOR_BGR2HSV)
            ground = hsv[300:520]
            neutral = float(((ground[:, :, 1] < 35) & (ground[:, :, 2] > 90)).mean())
            if neutral > 0.70:
                eligible.remove(index)
                rejected.append({"index": index, "reason": "neutral paved final foreground",
                                 "neutral_fraction": round(neutral, 3)})
    if not eligible:
        raise ValueError("No geometry-safe candidate view")
    model = SentenceTransformer(model_id, config_kwargs={"audio_config": None},
                                model_kwargs={"torch_dtype": torch.bfloat16 if torch.cuda.is_available()
                                              and torch.cuda.is_bf16_supported() else torch.float32},
                                device="cuda" if torch.cuda.is_available() else "cpu")
    if contacts is not None and not route_query:
        raise ValueError("A route query is required with contact sheets")
    inputs = [query] + ([route_query] if contacts is not None else [])
    for index in eligible:
        path = previews / f"{index + 1:04d}.png"
        with Image.open(path) as image:
            inputs.append({"image": image.convert("RGB").copy()})
    if contacts is not None:
        for index in eligible:
            path = contacts / f"{index + 1:04d}.png"
            with Image.open(path) as image:
                inputs.append({"image": image.convert("RGB").copy()})
    embeddings = model.encode(inputs, normalize_embeddings=True, show_progress_bar=False)
    query_count = 2 if contacts is not None else 1
    final_scores = np.asarray(embeddings[query_count:query_count + len(eligible)] @ embeddings[0], dtype=np.float32)
    route_scores = (np.asarray(embeddings[query_count + len(eligible):] @ embeddings[1], dtype=np.float32)
                    if contacts is not None else final_scores)
    combined = 0.65 * final_scores + 0.35 * route_scores if contacts is not None else final_scores
    ranking = sorted(({"index": index, "similarity": round(float(similarity), 5),
                       "final_similarity": round(float(final_score), 5),
                       "route_similarity": round(float(route_score), 5),
                       "geometry_score": plan[index]["score"]}
                      for index, similarity, final_score, route_score in zip(
                          eligible, combined, final_scores, route_scores)),
                     key=lambda item: (-item["similarity"], item["geometry_score"], item["index"]))
    return {"model": model_id, "query": query, "route_query": route_query, "ranked": ranking,
            "rejected": rejected,
            "selected_index": ranking[0]["index"], "selected": plan[ranking[0]["index"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--previews", type=Path, required=True)
    parser.add_argument("--contacts", type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--route-query")
    parser.add_argument("--avoid-pavement", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select_view(json.loads(args.plan.read_text()), args.previews, args.query,
                         contacts=args.contacts, route_query=args.route_query,
                         avoid_pavement=args.avoid_pavement,
                         model_id=os.getenv("TOUCHDOWN_EMBED_MODEL", "google/embeddinggemma-2"))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"selected_index": result["selected_index"],
                      "selected": result["selected"], "ranking": result["ranked"]}), flush=True)


if __name__ == "__main__":
    main()
