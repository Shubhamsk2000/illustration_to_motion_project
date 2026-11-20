"""Parse an episode script (Markdown) into chapters and scenes.

Format:
    # Episode title
    ## Chapter heading            -> a YouTube chapter
    A paragraph of narration.     -> one scene = one image
    > Image: what to draw          -> image note for that scene (not narrated)
    <!-- comments are ignored -->
"""

import re
from dataclasses import dataclass, field

WORDS_PER_MINUTE = 150  # typical documentary narration pace


@dataclass
class Scene:
    id: str                 # "001", "002", ...
    text: str               # narration for this scene
    chapter: str | None = None
    image_note: str = ""

    @property
    def words(self) -> list[str]:
        return self.text.split()


@dataclass
class Script:
    title: str
    scenes: list[Scene] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(s.words) for s in self.scenes)

    @property
    def chapters(self) -> list[str]:
        seen = []
        for s in self.scenes:
            if s.chapter and s.chapter not in seen:
                seen.append(s.chapter)
        return seen


def parse_script(markdown: str) -> Script:
    markdown = re.sub(r"<!--.*?-->", "", markdown, flags=re.S)
    title, chapter = "Untitled episode", None
    scenes: list[Scene] = []
    para: list[str] = []
    notes: list[str] = []

    def flush():
        nonlocal para, notes
        text = " ".join(" ".join(para).split())
        if text:
            scenes.append(Scene(f"{len(scenes) + 1:03d}", text, chapter, " ".join(notes).strip()))
        elif notes and scenes:  # a note right after a paragraph belongs to it
            scenes[-1].image_note = (scenes[-1].image_note + " " + " ".join(notes)).strip()
        para, notes = [], []

    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif line.startswith("## "):
            flush()
            chapter = line[3:].strip()
        elif line.startswith("# "):
            flush()
            title = line[2:].strip()
        elif line.startswith(">"):
            note = line.lstrip(">").strip()
            notes.append(re.sub(r"^image\s*:\s*", "", note, flags=re.I))
        else:
            para.append(line)
    flush()
    return Script(title, scenes)


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower()).strip("'")


SCRIPT_TEMPLATE = """# {title}

<!--
How to write this script
- One paragraph = one image on screen. Aim for 1-3 sentences (about 15-35 words, 6-14 seconds).
- "## Heading" lines start a YouTube chapter (YouTube needs at least 3, the first at 0:00).
- A line starting with "> Image:" describes the picture for that paragraph. It is not narrated;
  it goes into shotlist.md to help you generate the image.
- Then run:  motion-episode plan <this folder>
-->

## The morning

Before the sun rises over the camp, Marcus is already awake, pulling on a tunic still damp from yesterday's march.
> Image: Roman legionary in a dark tent at dawn, pulling on a red tunic, lantern light

## The march

By the third hour the column is on the road, twenty miles to cover before nightfall.
> Image: Long column of Roman soldiers marching on a stone road through hills, morning haze

## The night

At dusk the men gather around the fire, sharing bread, olives and stories from home.
> Image: Soldiers sitting around a campfire at night, warm firelight, stars above
"""
