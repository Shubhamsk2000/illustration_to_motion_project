"""motion-episode: script + narration + images -> finished YouTube episode.

    motion-episode new episodes/roman-soldier --title "A Day in the Life of a Roman Soldier"
    motion-episode plan episodes/roman-soldier        # after writing script.md: shot list + scene settings
    motion-episode voice episodes/roman-soldier       # listen to the cleaned-up narration
    motion-episode build episodes/roman-soldier --draft
    motion-episode build episodes/roman-soldier
"""

import argparse
import sys

from motion_studio.episode import audio
from motion_studio.episode.build import build, clean_voice
from motion_studio.episode.project import Episode


def cmd_new(args):
    ep = Episode(args.folder)
    ep.create(args.title or ep.dir.name.replace("-", " ").title())
    print(f"Created {ep.dir}/\n  script.md     write your script here (it starts with an example)\n"
          f"  episode.yaml  settings\n  images/       put scene images here\n"
          f"Next: write script.md, then run  motion-episode plan {ep.dir}")


def cmd_plan(args):
    ep = Episode(args.folder)
    r = ep.plan()
    print(f"{r['title']}\n  {r['scenes']} scenes, {r['words']} words, about {r['est_minutes']} min "
          f"(~{r['avg_scene_seconds']} s per image)")
    print(f"  chapters: {len(r['chapters'])}" + (f" ({', '.join(r['chapters'])})" if r["chapters"] else ""))
    if 0 < len(r["chapters"]) < 3:
        print("  ! YouTube needs at least 3 chapters to show them")
    print(f"  images: {r['images_found']}/{r['scenes']} found")
    if r["long_scenes"]:
        print(f"  ! long paragraphs (over 40 words, one image for 16 s+): {', '.join(r['long_scenes'])}"
              " - consider splitting them")
    if r["short_scenes"]:
        print(f"  ! very short paragraphs: {', '.join(r['short_scenes'])}")
    if r["dropped_overrides"]:
        print(f"  ! settings for scenes that no longer exist were dropped: {', '.join(r['dropped_overrides'])}")
    print(f"Wrote {ep.dir}/shotlist.md (what image to make for each scene) and episode.yaml")


def cmd_voice(args):
    ep = Episode(args.folder)
    out = clean_voice(ep, ep.settings().enhance_voice)
    dst = ep.export_dir / "narration_clean.wav"
    ep.export_dir.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(out.read_bytes())
    print(f"Cleaned narration: {dst} ({audio.duration(dst):.1f}s)")


def cmd_build(args):
    r = build(args.folder, draft=args.draft, allow_missing=args.allow_missing)
    print(f"\nDone in {r['build_seconds']}s -> {r['final']}")
    print(f"  {r['seconds']}s video, {r['scenes']} scenes ({r['segments_rendered']} rendered, "
          f"{r['segments_reused']} reused from the last build)")
    print(f"  script matched to the recording: {r['script_match'] * 100:.0f}%"
          + ("  ! low: check that the recording follows script.md" if r["script_match"] < 0.7 else ""))
    print(f"  subtitles.srt ({r['subtitle_cues']} cues), chapters.txt ({r['chapters']} chapters), timeline.json")
    for p in r["chapter_problems"]:
        print(f"  ! chapters: {p}")
    if r["missing_images"]:
        print(f"  ! placeholder cards for {len(r['missing_images'])} missing image(s)")
    if r["long_scenes"]:
        print(f"  ! scenes on screen over 16 s: {', '.join(r['long_scenes'])}")
    print(f"  music: {r['music'] or 'none'}")


def main():
    ap = argparse.ArgumentParser(prog="motion-episode", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new", help="create an episode folder with a script template")
    p.add_argument("folder")
    p.add_argument("--title")
    p.set_defaults(func=cmd_new)
    p = sub.add_parser("plan", help="read script.md, write shotlist.md and episode.yaml")
    p.add_argument("folder")
    p.set_defaults(func=cmd_plan)
    p = sub.add_parser("voice", help="write the cleaned-up narration to export/ to listen to")
    p.add_argument("folder")
    p.set_defaults(func=cmd_voice)
    p = sub.add_parser("build", help="render the finished episode")
    p.add_argument("folder")
    p.add_argument("--draft", action="store_true", help="fast 960x540 preview; missing images become cards")
    p.add_argument("--allow-missing", action="store_true", help="full quality, with cards for missing images")
    p.set_defaults(func=cmd_build)
    args = ap.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
