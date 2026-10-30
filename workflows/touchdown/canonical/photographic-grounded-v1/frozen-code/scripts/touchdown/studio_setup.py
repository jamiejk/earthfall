"""Build a saved photographic studio setup for any place, from Wikimedia Commons references.

The studio pipeline takes a setup: a destination, a subject photograph and two or more licensed
location photographs. Until now those were assembled by the web app. This makes the same setup from
the command line, so a new landmark can be run without touching the UI, and so the reference
selection is repeatable and recorded.

usage:
  studio_setup.py --place "Machu Picchu" --lat -13.1631 --lon -72.5450 \
      --region "Cusco" --country Peru --subject subject.jpg --out outputs/touchdown/studio/RUN
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import unicodedata
from pathlib import Path

from datetime import datetime, timezone
from typing import Literal

import httpx
from pydantic import BaseModel

from geoblock_core.place_photos import (USER_AGENT, _download, attribution, fetch_place_photos,
                                        find_place_photos, photo_report, pick_diverse)

ENDING = ("arms-open", "twirl", "relaxed")
FRAMING = ("three-quarter", "full-body", "head-and-shoulders")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug(title: str) -> str:
    """ASCII, lowercase, hyphenated: the destination id ends up in paths and URLs.

    Accents are folded rather than dropped, so Chichén Itzá and Aït Benhaddou keep a readable id
    instead of losing a letter.
    """
    folded = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return "-".join("".join(c if c.isalnum() else " " for c in folded.lower()).split())


def download_picked(place: str, latitude: float, longitude: float, page_ids: tuple[int, ...],
                    folder: Path, radius_km: float = 2.0) -> dict:
    """Download exactly the Commons pages a reviewer chose, in the manifest shape the setup needs.

    Reference selection is a visual judgement: Commons has 52 eligible photos of a famous summit and
    nearly all of them have tourists in the foreground, which then get invented into the arrival. The
    ranking cannot tell a crowded terrace from an empty one, so a reviewer picks page ids and this
    records the choice. Same manifest shape as ``fetch_place_photos`` so everything downstream is
    identical.
    """
    folder.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=90, follow_redirects=True) as client:
        candidates = find_place_photos(place, latitude, longitude, radius_km=radius_km,
                                       purpose="landscape", client=client)
        by_id = {c["page_id"]: c for c in candidates}
        missing = [p for p in page_ids if p not in by_id]
        if missing:
            raise ValueError(f"page ids not within {radius_km:g} km of {place}: {missing}")
        selected = [_download(client, by_id[p], folder, 25 * 2 ** 20) for p in page_ids]
    return {"place": place, "latitude": latitude, "longitude": longitude, "radius_km": radius_km,
            "source": "Wikimedia Commons",
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "purpose": "landscape",
            "picked_by": "operator", "selected": selected, "candidates": candidates}


SHORTLIST = 12
USABLE_VIEWS = ("landmark_exterior", "aerial_of_landmark")


class ReferenceScreen(BaseModel):
    shows_named_landmark: bool
    view: Literal["landmark_exterior", "aerial_of_landmark", "interior", "detail_or_object", "other_place", "unclear"]
    people_prominent: bool
    natural_daylight: bool
    fisheye_or_distorted: bool
    evidence: str


def usable(screen: dict) -> bool:
    return (screen["shows_named_landmark"] and screen["view"] in USABLE_VIEWS and not screen["people_prominent"] and screen["natural_daylight"]
            and not screen["fisheye_or_distorted"])


def screened_photos(place: str, region: str, country: str, latitude: float, longitude: float, count: int,
                    folder: Path, judge, radius_km: float = 2.0, exclude: tuple[int, ...] = ()) -> dict:
    """Download the metadata-ranked shortlist and keep the first photos the local vision model accepts.

    The Commons ranking reads titles and coordinates only: for Chambord its top picks were a carriage
    under a tent and a stained-glass monogram, both titled "Chambord". Each shortlisted photo is shown
    to the model on its own and kept only when it shows the landmark from outside or above with no
    prominent person, in natural daylight (Chartres' screen took a night light-show projection). ``judge(path, prompt, output)`` returns a ReferenceScreen dict.
    """
    folder.mkdir(parents=True, exist_ok=True)
    prompt = (f"Is this a photograph of {place}, {region}, {country}? shows_named_landmark is true only when that "
              "landmark itself is the main subject; false for another building nearby (a library, museum, church, "
              "hotel or street) that merely shares its name or town. Classify the view: landmark_exterior (the "
              "landmark's outside and the ground around it), aerial_of_landmark (from above), interior, "
              "detail_or_object (a close detail, artwork, vehicle, sign, exhibit or object), other_place, or "
              "unclear. people_prominent is true when any person is large or central in the frame. natural_daylight is "
              "false for night, dusk floodlighting, light shows, projections or coloured stage lighting. "
              "fisheye_or_distorted is true for fisheye or ultra-wide bulging lines, or a view through glass, "
              "window bars or lattice.")
    screened, selected = [], []
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as client:
        candidates = find_place_photos(place, latitude, longitude, radius_km=radius_km,
                                       purpose="landscape", client=client)
        for item in pick_diverse([c for c in candidates if c["page_id"] not in exclude], SHORTLIST):
            try:
                item = _download(client, item, folder, 25 * 2 ** 20)
            except (httpx.HTTPError, OSError, ValueError) as exc:
                print(f"Skipping {item['title']}: {exc}")
                continue
            answer = judge(Path(item["local_path"]), prompt, folder / "screen" / f"{item['page_id']}.json")
            screened.append({"page_id": item["page_id"], "title": item["title"], **answer, "accepted": usable(answer)})
            if usable(answer):
                selected.append(item)
                if len(selected) == count:
                    break
            else:
                Path(item["local_path"]).unlink(missing_ok=True)
    if len(selected) < 2:
        raise ValueError(f"The vision screen accepted {len(selected)} of {len(screened)} shortlisted Commons "
                         f"photos of {place}; the studio needs two. See {folder / 'screen'}.")
    manifest = {"place": place, "latitude": latitude, "longitude": longitude, "radius_km": radius_km,
                "source": "Wikimedia Commons", "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "landscape", "picked_by": "vision screen", "excluded_page_ids": list(exclude),
                "screened": screened, "selected": selected, "candidates": candidates}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (folder / "ATTRIBUTION.md").write_text(attribution(selected))
    return manifest


def vision_judge(path: Path, prompt: str, output: Path) -> dict:
    """The pinned local checker the build uses; the caller holds its runtime."""
    from touchdown.studio_brief import chat
    from touchdown.studio_build import VISION_URL
    return chat(VISION_URL, [(path, "PHOTO")], prompt, ReferenceScreen, output)


def setup_from(manifest: dict, subject: dict, destination: dict, *, ending: str,
               framing: str, wearing: str = "") -> dict:
    """Assemble the saved setup from a downloaded reference manifest and a subject photograph.

    Kept separate from the download so the mapping can be tested without the network, and so a
    reviewer can rewrite the reference list and rebuild the setup from the manifest on disk.
    """
    if len(manifest["selected"]) < 2:
        raise ValueError(f"{len(manifest['selected'])} usable references; the studio needs two")
    references = []
    for index, item in enumerate(manifest["selected"]):
        references.append({
            "name": f"location-{index}.jpg",
            "source": item.get("source", "Wikimedia Commons"),
            "page_id": item["page_id"],
            "title": item["title"],
            "artist": item.get("artist", ""),
            "license": item.get("license", ""),
            "license_url": item.get("license_url", ""),
            "url": item.get("commons_page") or item.get("image_url", ""),
            "source_sha256": item.get("source_sha256", ""),
            "sha256": item["sha256"],
        })
    body = {"destination": destination, "ending": ending, "framing": framing, "wearing": wearing,
            "subject": {"name": subject["name"], "sha256": subject["sha256"]},
            "references": references}
    # The saved setup carries the full attribution the worker validates by name and hash; the id is
    # taken over only what decides the generated content, so re-fetching a file cannot change it.
    identity = {**{k: body[k] for k in ("destination", "ending", "framing", "wearing", "subject")},
                "references": [{k: r[k] for k in ("page_id", "title", "sha256")} for r in references]}
    identifier = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:32]
    return {"id": identifier, **body}


def build(place: str, latitude: float, longitude: float, region: str, country: str,
          subject_path: Path, root: Path, *, count: int = 2, ending: str = "arms-open",
          framing: str = "three-quarter", wearing: str = "", exclude: tuple[int, ...] = (),
          pick: tuple[int, ...] = (), screen: bool = True) -> dict:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    subject_name = f"subject{subject_path.suffix.lower() or '.jpg'}"
    target = inputs / subject_name
    if target.resolve() != subject_path.resolve():
        shutil.copyfile(subject_path, target)
    subject = {"name": subject_name, "sha256": digest(target)}

    if pick:
        manifest = download_picked(place, latitude, longitude, pick, inputs)
    elif screen:
        from touchdown.studio_build import runtime
        with runtime("vision", root):
            manifest = screened_photos(place, region, country, latitude, longitude, count, inputs,
                                       vision_judge, exclude=exclude)
    else:
        manifest = fetch_place_photos(place, latitude, longitude, count, inputs,
                                      purpose="landscape", exclude=exclude)
    print(photo_report(manifest))
    # The studio reads references by their setup name, so the downloads are renamed into place.
    for index, item in enumerate(manifest["selected"]):
        source = Path(item["local_path"])
        renamed = inputs / f"location-{index}.jpg"
        if source.resolve() != renamed.resolve():
            shutil.copyfile(source, renamed)
            source.unlink(missing_ok=True)
        item["sha256"] = digest(renamed)
        item["local_path"] = str(renamed.resolve())

    destination = {"id": slug(place), "title": place, "region": region, "country": country,
                   "latitude": latitude, "longitude": longitude}
    setup = setup_from(manifest, subject, destination, ending=ending, framing=framing, wearing=wearing)
    (root / "setup.json").write_text(json.dumps(setup, indent=2, ensure_ascii=False) + "\n")
    return setup


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--place", required=True)
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--region", default="")
    p.add_argument("--country", default="")
    p.add_argument("--subject", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--references", type=int, default=2)
    p.add_argument("--ending", choices=ENDING, default="arms-open")
    p.add_argument("--framing", choices=FRAMING, default="three-quarter")
    p.add_argument("--wearing", default="")
    p.add_argument("--pick", type=int, nargs="*", default=[],
                   help="exact Commons page ids to use, in order, bypassing the ranking")
    p.add_argument("--exclude", type=int, nargs="*", default=[],
                   help="Commons page ids rejected on review; the next best photos take their place")
    p.add_argument("--no-screen", action="store_true",
                   help="take the metadata ranking as it is, without the local vision screen")
    a = p.parse_args()
    setup = build(a.place, a.lat, a.lon, a.region, a.country, a.subject.resolve(),
                  a.out.resolve(), count=a.references, ending=a.ending, framing=a.framing,
                  wearing=a.wearing, exclude=tuple(a.exclude), pick=tuple(a.pick),
                  screen=not a.no_screen)
    print(f"\nsetup {setup['id']} written to {a.out / 'setup.json'}")
    print(f"  {len(setup['references'])} references, subject {setup['subject']['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
