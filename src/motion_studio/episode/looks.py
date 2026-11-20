"""Pick a look for a scene from its words: the "> Image:" note if there is one, otherwise the narration.

Simple keyword rules, checked in order (the first match wins). Any scene can override it in episode.yaml.
"""

import re

RULES: list[tuple[str, tuple[str, ...]]] = [
    ("campfire", ("fire", "campfire", "bonfire", "torch", "torches", "torchlight", "candle", "candles", "candlelight",
                  "hearth", "flames", "flame", "forge", "lantern", "lanterns", "lamp", "lamplight", "embers")),
    ("storm", ("rain", "rainy", "storm", "stormy", "thunder", "lightning", "downpour")),
    ("winter", ("snow", "snowy", "winter", "frozen", "frost", "blizzard", "ice", "icy")),
    ("battle", ("battle", "battlefield", "siege", "charge", "clash", "war", "skirmish", "smoke")),
    ("misty", ("fog", "foggy", "mist", "misty", "haze", "hazy", "dawn", "daybreak")),
    ("sunlit", ("sun", "sunlight", "sunny", "sunrise", "sunset", "sunbeam", "sunbeams", "noon", "desert")),
    ("portrait", ("close-up", "closeup", "portrait", "face", "eyes")),
]


def suggest_look(text: str, note: str = "") -> tuple[str | None, str | None]:
    """(preset, the word that triggered it), or (None, None) for the default look."""
    source = (note or text).lower()
    words = set(re.findall(r"[a-z][a-z-]*", source))
    for preset, keys in RULES:
        for k in keys:
            if k in words:
                return preset, k
    return None, None
