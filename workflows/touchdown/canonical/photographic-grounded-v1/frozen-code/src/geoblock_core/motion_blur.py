"""Motion blur as a final post step on an Earth zoom (``--motion-blur targeted``), before any upscale.

The swoop and the landing smear in LTX (the camera moves several picture widths a second there); with blur
they read as speed. The cloud and the eye-level hold stay sharp. Jamie chose this "targeted" blur on Twin
Peaks r46: frames are motion-interpolated to 4 subframes each, then averaged over a ~180 degree shutter
from the deck exit to just past the arrival, and a ~360 degree one over the landing, each faded in and out
over RAMP_FRAMES. All ffmpeg (minterpolate, tmix, blend), on the CPU: about 5 minutes at 960x544.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

RAMP_FRAMES = 4
SUBFRAMES = 4  # minterpolate to 4x the frame rate


def windows(frames: int, exit_frame: float, arrive_frame: int) -> dict:
    """Frame ranges (fade-in start, fade-out end) of the light and strong blur for a shot of ``frames``: light
    from the punch-through just before the deck exit to just past the arrival, strong over the landing's last
    fast stretch (r46: light f24-116, strong f80-104 with a 108 arrival)."""
    light_in = max(0, round(exit_frame) - 3 - RAMP_FRAMES)
    light = (light_in, min(frames - 1, arrive_frame + 4 + RAMP_FRAMES))
    strong = (max(light_in, arrive_frame - 28), arrive_frame - 4)
    return {"light": light, "strong": strong}


def weight_expr(window: tuple) -> str:
    """ffmpeg expression in the frame number N: 0 outside ``window``, ramping to 1 over RAMP_FRAMES at each end."""
    start, end = window
    return (f"clip((N-{start})/{RAMP_FRAMES}\\,0\\,1)*clip(({end}-N)/{RAMP_FRAMES}\\,0\\,1)")


def filter_graph(fps: float, spans: dict) -> str:
    """The filtergraph: the sharp frames, a 2-subframe average (~180 degrees) and a 4-subframe one (~360),
    each taken at the same subframe so the three line up, blended on the frame number."""
    rate = fps * SUBFRAMES
    return (
        f"[0:v]setpts=N/({fps}*TB),split=2[base][m];"
        f"[m]minterpolate=fps={rate:g}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,split=2[m1][m2];"
        f"[m1]tmix=frames=2,select='eq(mod(n\\,{SUBFRAMES})\\,1)',setpts=N/({fps}*TB)[light];"
        f"[m2]tmix=frames=4,select='eq(mod(n\\,{SUBFRAMES})\\,2)',setpts=N/({fps}*TB)[strong];"
        f"[base][light]blend=all_expr='A+(B-A)*{weight_expr(spans['light'])}'[lit];"
        f"[lit][strong]blend=all_expr='A+(B-A)*{weight_expr(spans['strong'])}'[out]"
    )


def apply(video: Path, destination: Path, frames: int, exit_frame: float, arrive_frame: int,
          fps: float = 24.0) -> Path:
    """Write ``video`` with the targeted motion blur to ``destination``."""
    spans = windows(frames, exit_frame, arrive_frame)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-filter_complex", filter_graph(fps, spans),
                    "-map", "[out]", "-frames:v", str(frames), "-c:v", "libx264", "-crf", "14",
                    "-pix_fmt", "yuv420p", "-r", f"{fps:g}", str(destination)], check=True)
    return destination
