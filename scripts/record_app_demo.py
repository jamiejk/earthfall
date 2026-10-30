"""Record the Touchdown app for the demo video, with Playwright so it stays programmatic.

    python scripts/record_app_demo.py --out docs/demo/app-recording.webm

It opens the app, picks the flagship place, uploads a photograph, types what the person is
wearing, starts the build, and watches the stages until the clip is ready. The recording is
a screen capture of a headless browser: 1920x1080, 24 fps.

It records against a *running* app. With the mock build (``TOUCHDOWN_BUILD_CMD`` pointing at
``app/worker/fake_build.py``) the whole thing takes a minute and needs no GPU, which is what
this script is for; against the real pipeline the build takes longer than the recording.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

SUBJECT = Path(__file__).resolve().parents[1] / "shots" / "subjects" / "test-subject-red-coat.jpg"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("docs/demo/app-recording.webm"))
    p.add_argument("--url", default="http://127.0.0.1:8020/")
    p.add_argument("--place", default="Monument Valley overlook")
    p.add_argument("--wearing", default="a long dark green raincoat")
    p.add_argument("--photo", type=Path, default=SUBJECT)
    p.add_argument("--seconds", type=float, default=45.0, help="how long to watch the build")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=["--force-device-scale-factor=1"])
        context = browser.new_context(viewport={"width": args.width, "height": args.height},
                                      record_video_dir=str(args.out.parent),
                                      record_video_size={"width": args.width, "height": args.height})
        page = context.new_page()
        page.goto(args.url, wait_until="networkidle")
        page.wait_for_timeout(2500)

        # the place list: the flagship is preselected, so the click is the point
        page.get_by_role("option", name=args.place).click()
        page.wait_for_timeout(1200)

        # the photograph and what they are wearing
        page.set_input_files("input[type=file]", str(args.photo))
        wearing = page.locator("#wearing")
        if wearing.count():
            wearing.fill(args.wearing)
        page.wait_for_timeout(1200)

        # start it
        button = page.get_by_role("button", name=re.compile("make my touchdown|build|start", re.I))
        if button.count():
            button.first.click()
        else:
            page.get_by_role("button").last.click()
        page.wait_for_timeout(1500)

        # watch the stages
        deadline = time.time() + args.seconds
        while time.time() < deadline:
            page.wait_for_timeout(1000)
        page.wait_for_timeout(1500)

        video = page.video
        context.close()
        browser.close()
        if video:
            saved = video.path()
            saved.rename(args.out)
            print(args.out)


if __name__ == "__main__":
    main()
