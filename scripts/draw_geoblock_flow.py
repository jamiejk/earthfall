"""Render a compact visual guide to the working ComfyUI prototype."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/geoblock_prototype_flow.png"
FONT = "/usr/share/fonts/google-noto/NotoSans-Regular.ttf"


def font(size, bold=False):
    name = "NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf"
    return ImageFont.truetype(str(Path(FONT).with_name(name)), size)


def main():
    canvas = Image.new("RGB", (1900, 750), "#111922")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 42), "GeoBlock to H3  |  Worm's Head prototype", fill="#F4F7FA", font=font(36, True))
    draw.text((72, 96), "A working ComfyUI preparation flow · place and shot in, control video and checked photos out",
              fill="#AAC0CF", font=font(19))
    cards = [
        (70, "1  Resolve place", ["Worm's Head, Rhossili", "Coverage radius: 2.5 km", "Exact coordinates available"], "#427C8F"),
        (520, "2  Terrain scene", ["Pinned LiDAR asset", "Connected headland", "Lookout preview"], "#598567"),
        (970, "3  Plan shot", ["Wide turn + slow return", "7.3 s = 175 frames", "Camera route preview"], "#917B4E"),
        (1420, "4  Build references", ["Retimed control MP4", "2 licensed place photos", "H3 prompt + frame count"], "#7768A4"),
    ]
    for left, title, lines, colour in cards:
        draw.rounded_rectangle((left, 160, left + 410, 350), radius=20, fill="#1B2732", outline=colour, width=3)
        draw.text((left + 25, 183), title, fill="#F6F8FA", font=font(23, True))
        for index, line in enumerate(lines):
            draw.text((left + 28, 240 + index * 30), line, fill="#D3E1E8", font=font(18))
    for x in (480, 930, 1380):
        draw.line((x + 5, 256, x + 34, 256), fill="#79B6C8", width=5)
        draw.polygon([(x + 35, 256), (x + 20, 247), (x + 20, 265)], fill="#79B6C8")

    images = [
        (ROOT / "outputs/B_lidar_full_2m/sea_preview_lookout.png", 520, "Terrain lookout"),
        (ROOT / "data/reference_images/worms_head/commons_115697390.jpg", 970, "Place photo 1"),
        (ROOT / "data/reference_images/worms_head/commons_133768741.jpg", 1420, "Place photo 2"),
    ]
    for path, left, label in images:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((410, 240))
        bg = Image.new("RGB", (410, 240), "#0D141A")
        bg.paste(image, ((410 - image.width) // 2, (240 - image.height) // 2))
        canvas.paste(bg, (left, 395))
        draw.rectangle((left, 395, left + 410, 635), outline="#385467", width=2)
        draw.text((left, 653), label, fill="#DDEAF1", font=font(19, True))
    draw.rounded_rectangle((70, 425, 465, 630), radius=18, fill="#16212B", outline="#427C8F", width=2)
    draw.text((96, 450), "What the user edits", fill="#F6F8FA", font=font(22, True))
    draw.text((96, 501), "Place · area · route · duration", fill="#D3E1E8", font=font(18))
    draw.text((96, 536), "Visual style · sea motion", fill="#D3E1E8", font=font(18))
    draw.text((96, 591), "H3 generation is the next node", fill="#A8BDCB", font=font(17))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
