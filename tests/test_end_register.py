import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from touchdown.end_register import H, W, fit  # noqa: E402


def test_shifted_scaled_skyline_is_recovered():
    u = np.arange(W)
    guide = 300 + 40 * np.sin(u / 70.0) + 25 * np.exp(-((u - 600) / 40.0) ** 2)  # ridges and one peak
    s, x0, y0 = 1.2, 64.0, 30.0
    # the end still shows the guide zoomed out: end pixel (x0 + u / s, y0 + v / s) is guide pixel (u, v)
    end = np.full(W, np.nan)
    xe = x0 + u / s
    end_x = np.arange(W)
    inside = (end_x >= xe[0]) & (end_x <= xe[-1])
    end[inside] = y0 + np.interp(end_x[inside], xe, guide) / s
    end[440:520] = np.nan  # the person hides the skyline
    got = fit(guide, end, scales=np.arange(1.0, 1.3501, 0.01), step=2, height=H, inside_only=True)
    assert abs(got["scale"] - s) <= 0.011
    assert abs(got["x0"] - x0) <= 3
    assert abs(got["y0"] - y0) <= 2
    assert got["residual_px"] < 1.5
