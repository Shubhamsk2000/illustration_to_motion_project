"""Camera moves as functions of normalized time t in [0, 1].

Each move returns (zoom, dx, dy): zoom >= 1 is extra magnification, dx/dy shift the picture as fractions of the
output width/height. One-way moves ease in and out (documentary style); loop moves end where they start.
"""

import math

ONE_WAY_MOVES = ["push_in", "pull_out", "pan_left", "pan_right", "tilt_up", "tilt_down", "static"]
LOOP_MOVES = ["drift", "breathe", "static"]
ALL_MOVES = sorted(set(ONE_WAY_MOVES + LOOP_MOVES))

# Order used by --move auto so consecutive scenes don't repeat the same move.
AUTO_SEQUENCE = ["push_in", "pan_right", "pull_out", "pan_left", "push_in", "tilt_up"]

ZOOM = 0.08  # push-in/pull-out zoom at strength 1.0
PAN = 0.05   # total pan travel at strength 1.0 (fraction of frame)


def ease(t: float) -> float:
    """Smooth start and stop."""
    t = min(1.0, max(0.0, t))
    return 0.5 - 0.5 * math.cos(math.pi * t)


def camera_at(move: str, t: float, strength: float = 1.0) -> tuple[float, float, float]:
    z, p = ZOOM * strength, PAN * strength
    e = ease(t)
    loop = 2 * math.pi * t
    if move == "static":
        return 1.0, 0.0, 0.0
    if move == "push_in":
        return 1.0 + z * e, 0.0, 0.0
    if move == "pull_out":
        return 1.0 + z * (1.0 - e), 0.0, 0.0
    if move == "pan_right":  # camera moves right, so the picture slides left
        return 1.0 + z * 0.25, p * (0.5 - e), 0.0
    if move == "pan_left":
        return 1.0 + z * 0.25, p * (e - 0.5), 0.0
    if move == "tilt_up":  # camera moves up, picture slides down
        return 1.0 + z * 0.25, 0.0, p * (e - 0.5)
    if move == "tilt_down":
        return 1.0 + z * 0.25, 0.0, p * (0.5 - e)
    if move == "drift":
        return 1.0 + z * 0.25 * (0.5 - 0.5 * math.cos(loop)), p * 0.3 * math.sin(loop), p * 0.15 * math.cos(loop)
    if move == "breathe":
        return 1.0 + z * 0.5 * (0.5 - 0.5 * math.cos(loop)), 0.0, 0.0
    raise ValueError(f"Unknown camera move '{move}'. Choose from: {', '.join(ALL_MOVES)}")


def auto_move(index: int, loop: bool = False) -> str:
    if loop:
        return "drift" if index % 2 == 0 else "breathe"
    return AUTO_SEQUENCE[index % len(AUTO_SEQUENCE)]
