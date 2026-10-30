"""The Earth zoom's smooth ride, shared by the route planner and the Blender script (plain Python, no numpy)."""

from __future__ import annotations

# "smooth-swoop" is low-swoop with a smooth ride (SMOOTH_TEMPOS): Twin Peaks r44 read as bumpy. The camera came down
# at ~0.44 m a frame and stopped dead at the arrival frame, then the push-in started again from rest (frame
# difference 12.7 -> 2.0 over f108-111); and below 3 km it followed the ground under it, so it rose and fell
# with every hill it passed over. It brakes over ARRIVAL_EASE_FRAMES into a push-in already moving, and rides a
# smoothed upper envelope of the ground (ride_floor) instead of the ground itself.
SMOOTH_TEMPOS = ("smooth-swoop",)
ARRIVAL_EASE_FRAMES = 36
RIDE_FLOOR_FRAMES = 12


def ride_floor(values, radius: int) -> list:
    """A smooth upper envelope of ``values`` (one per frame): the running maximum over ``radius`` frames each way,
    then two box averages of half that radius. Every frame stays at or above its own value (each average only
    takes maxima of windows containing it), and the curve has no corners. Plain Python: Blender imports it too."""
    n, radius = len(values), max(0, int(radius))
    if n == 0 or radius == 0:
        return list(values)
    peak = [max(values[max(0, i - radius):i + radius + 1]) for i in range(n)]
    half = max(1, radius // 2)
    for _ in range(2):
        peak = [sum(peak[max(0, i - half):i + half + 1]) / len(peak[max(0, i - half):i + half + 1])
                for i in range(n)]
    return peak
