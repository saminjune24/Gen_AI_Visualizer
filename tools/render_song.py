"""Run the visualizer notebook on one audio file, without editing the notebook.

Usage (from the repo root):
    .venv/bin/python tools/render_song.py sample_inputs/twenty_one_pilots_drag_path.mp3 --start 105
    .venv/bin/python tools/render_song.py sample_inputs/naika_barely_barely.mp3 --start 28 --vibe dark
    .venv/bin/python tools/render_song.py sample_inputs/demo_track.wav --style modes

Everything for the song lands in outputs/<song name>/ (videos, images, figures, tables), plus an executed
copy of the notebook (notebook_run.ipynb) so you can scroll through every plot for that song.
"""
import argparse
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VIBES = ["party", "happy", "chill", "melancholic", "dark", "epic", "bittersweet"]


def slugify(text):
    # Same rule as the notebook, so the output folder names match.
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="path to a .wav / .mp3 / .flac file")
    parser.add_argument("--start", type=float, default=0.0, help="clip start in seconds (the clip is 30 s long)")
    parser.add_argument("--vibe", choices=VIBES, help="force a vibe instead of detecting it")
    parser.add_argument("--style", choices=["psychedelic", "modes"], default="psychedelic", help="visual style")
    args = parser.parse_args()

    audio = args.audio.resolve()
    if not audio.exists():
        sys.exit(f"Audio file not found: {audio}")
    # The notebook runs from the repo root, so pass a path relative to it when possible
    audio_arg = str(audio.relative_to(ROOT)) if audio.is_relative_to(ROOT) else str(audio)

    env = os.environ.copy()
    env.update({"AUDIO_PATH": audio_arg, "CLIP_START": str(args.start), "VISUAL_STYLE": args.style})
    if args.vibe:
        env["VIBE_OVERRIDE"] = args.vibe

    out_dir = ROOT / "outputs" / slugify(audio.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {audio.name} (clip {args.start:.0f}-{args.start + 30:.0f} s) -> {out_dir.relative_to(ROOT)}/")
    # Call nbconvert through this Python (the project venv), not whatever `jupyter` is first on PATH
    subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
                    "--ExecutePreprocessor.timeout=3600", str(ROOT / "genai_audio_visualizer.ipynb"),
                    "--output-dir", str(out_dir), "--output", "notebook_run"],
                   check=True, env=env, cwd=ROOT)
    print(f"Done. Videos: {(out_dir / 'videos').relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
