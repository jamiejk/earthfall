#!/usr/bin/env bash
# Cut the two-minute demo video from the assets in docs/demo and app/demo-clips.
#
#   bash scripts/make_demo_video.sh [out.mp4]
#
# The pieces:
#   1. a hook: the best clip, first
#   2. the app, recorded with Playwright (scripts/record_app_demo.py)
#   3. three finished clips, different places
#   4. the proof: the real NAIP photograph beside the same place in the clip
#   5. the end card
#
# Everything is 1920x1080 at 24 fps, the length the competition asks for (under two minutes).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/docs/demo/touchdown-demo.mp4}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

FONT="/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf"
FPS=24
W=1920
H=1080

title() {  # text | size | x | y
  local esc
  esc="$(printf '%s' "$1" | sed 's/:/\\:/g')"
  printf "drawtext=fontfile=%s:text='%s':fontcolor=white:fontsize=%s:box=1:boxcolor=black@0.55:boxborderw=14:x=%s:y=%s" \
      "$FONT" "$esc" "$2" "$3" "$4"
}

if [ ! -f "$ROOT/docs/demo/app-recording.webm" ]; then
  echo "record the app first: python scripts/record_app_demo.py --out docs/demo/app-recording.webm" >&2
  exit 1
fi

# 1. the hook: the strongest clip, ten seconds
HOOK="${TOUCHDOWN_HOOK:-$ROOT/docs/demo/hook.mp4}"
if [ ! -f "$HOOK" ]; then
  echo "no hook clip at $HOOK (set TOUCHDOWN_HOOK)" >&2
  exit 1
fi
ffmpeg -y -v error -ss 1.4 -t 10 -i "$HOOK" -vf \
  "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,$(title 'Touchdown' 64 60 60)" \
  -r $FPS -c:v libx264 -crf 20 -pix_fmt yuv420p "$WORK/1-hook.mp4"

# 2. the app: the recording, cropped to the column the app draws in
ffmpeg -y -v error -t 32 -i "$ROOT/docs/demo/app-recording.webm" -vf \
  "crop=1180:1000:210:0,scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,$(title 'the app' 44 60 940)" \
  -r $FPS -c:v libx264 -crf 21 -pix_fmt yuv420p "$WORK/2-app.mp4"

# 3. the clips, each starting a few seconds in: the first seconds are the dive through the haze
i=3
for clip in "$ROOT"/app/demo-clips/monument-utah.mp4 "$ROOT"/app/demo-clips/grand-mather-point.mp4 \
            "$ROOT"/app/demo-clips/horseshoe-arizona.mp4; do
  [ -f "$clip" ] || continue
  ffmpeg -y -v error -ss 2.0 -i "$clip" -vf \
    "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080" \
    -r $FPS -c:v libx264 -crf 20 -pix_fmt yuv420p "$WORK/$i-clip.mp4"
  i=$((i + 1))
done

# 4. the proof: the real photograph beside the same place in the clip
if [ -f "$ROOT/docs/demo/real-vs-made.jpg" ]; then
  ffmpeg -y -v error -loop 1 -i "$ROOT/docs/demo/real-vs-made.jpg" -t 6 -vf \
    "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,$(title 'the ground is measured, not drawn' 52 60 60)" \
    -r $FPS -c:v libx264 -crf 20 -pix_fmt yuv420p "$WORK/9-proof.mp4"
fi

# 5. the end card
ffmpeg -y -v error -f lavfi -i "color=c=0x121418:s=1920x1080:d=7" -vf \
  "$(title 'Touchdown' 84 80 380),$(title 'Earth Zoom, but the place is real' 44 60 500),$(title 'github.com/jamiejk/touchdown' 34 60 560)" \
  -r $FPS -c:v libx264 -crf 20 -pix_fmt yuv420p "$WORK/10-end.mp4"

# the proof's caption is longer than one drawtext line, so it is set here
if [ -f "$WORK/9-proof.mp4" ]; then
  ffmpeg -y -v error -i "$WORK/9-proof.mp4" -vf \
    "drawtext=fontfile=$FONT:text='the real NAIP photograph, and the same landforms in the clip':fontcolor=white:fontsize=38:box=1:boxcolor=black@0.55:boxborderw=14:x=60:y=940" \
    -c:v libx264 -crf 20 -pix_fmt yuv420p "$WORK/9-proof-captioned.mp4"
  mv "$WORK/9-proof-captioned.mp4" "$WORK/9-proof.mp4"
fi

# the pieces in the order they play: the order is explicit, not a glob (a glob would sort "10-end" before "2-app")
: > "$WORK/list.txt"
for piece in 1-hook 2-app 3-clip 4-clip 5-clip 6-clip 7-clip 8-clip 9-proof 10-end; do
  [ -f "$WORK/$piece.mp4" ] && printf "file '%s'\n" "$WORK/$piece.mp4" >> "$WORK/list.txt"
done
ffmpeg -y -v error -f concat -safe 0 -i "$WORK/list.txt" -c copy "$OUT"
echo "$OUT"
ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT"
