"""Builds genai_audio_visualizer.ipynb from the cells below.

Editing the notebook directly in Jupyter is fine too; this script just keeps the
cell sources readable in a diff. Run:  .venv/bin/python build_notebook.py
"""
import nbformat as nbf

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# =============================================================================
# 1. Overview
# =============================================================================
md(r"""
# Section-Aware Procedural GenAI Audio Visualizer

**Pipeline:** `audio → features → sections → prompts → generated images → control timeline → animation → final video → baseline → evaluation`

This notebook turns a short music clip into an animated music visualizer that combines two parts:

| Part | What it does |
|---|---|
| **Generative** | A pretrained text-to-image diffusion model (SD-Turbo) creates one abstract image **per analysis section** of the song. Each prompt is built from that section's measured audio statistics. |
| **Procedural** | Python (NumPy + OpenCV) splits each generated image into parts (core, inner star, outer star, line art, light spots) and moves **each part** with a different musical signal: beats, low/mid/high frequency energy, RMS energy, onset strength, spectral brightness and the separated **instrument stems** (drums, bass, vocals, other). |
| **Vibe** | A pretrained audio-text model (CLAP) guesses the song's overall vibe (party, calm, dark, dreamy, epic), which sets the theme and palette of every prompt. |

### Research question
> Can audio-derived features be used to guide generative artwork and procedural animation to create a visualization that feels more synchronized and connected to the music than a non-audio-conditioned visualizer?

### Hypothesis
> The audio-conditioned visualizer will receive higher ratings for synchronization and perceived connection to the music than the non-audio-conditioned baseline.

To test this, the notebook renders **two videos** that share the same song, the same generated images and the same effect types:

1. **Audio-conditioned visualizer**: motion is driven by the extracted audio features.
2. **Baseline**: the same effects are driven by preset, fixed-rate rules that never look at the audio.

### What this notebook deliberately does *not* do
No frame-by-frame diffusion video, no real-time rendering, no model training, no animated people, no 3D. Images are generated once (3–4 total) and cached;
all motion comes from warping the parts of those images.
""")

# =============================================================================
# 2. Setup
# =============================================================================
md(r"""
## 2. Setup and imports

Runs locally (project `.venv`, see README) or on Google Colab. On Colab, the cell below installs the packages that are not pre-installed.
No system FFmpeg is needed: `imageio-ffmpeg` ships its own FFmpeg binary.
""")

code(r"""
import sys

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    # Colab already has torch, numpy, pandas, matplotlib, opencv.
    %pip install -q librosa soundfile imageio-ffmpeg diffusers transformers accelerate
""")

code(r"""
import hashlib
import json
import os
import re
import subprocess
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
from scipy.ndimage import maximum_filter1d, uniform_filter1d

import librosa
import soundfile as sf
import cv2
import imageio_ffmpeg
from PIL import Image
from IPython.display import Audio, Video, display

warnings.filterwarnings("ignore", category=FutureWarning)

# Pick the best available device for image generation: NVIDIA GPU -> Apple GPU -> CPU.
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
except ImportError:
    torch, DEVICE = None, "cpu"

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()   # path to the bundled ffmpeg binary

# One consistent colour per series across all plots (fixed order, never cycled).
COLORS = {"rms": "#2a78d6", "bass": "#eb6834", "mid": "#1baf7a", "high": "#eda100",
          "centroid": "#e87ba4", "onset": "#6250d6", "beat": "#52514e", "baseline": "#e34948",
          "drums": "#6250d6", "bass_stem": "#eb6834", "other": "#1baf7a", "vocals": "#e87ba4"}
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

print(f"Python {sys.version.split()[0]} | librosa {librosa.__version__} | OpenCV {cv2.__version__}")
print(f"Image-generation device: {DEVICE}")
""")

# =============================================================================
# 3. Configuration
# =============================================================================
md(r"""
## 3. Configuration

**Every adjustable parameter lives in this one cell**, so the rest of the notebook contains no hidden magic numbers.
The visual coefficients in `VIS` are *tuning parameters* chosen by eye, not scientific constants.
""")

code(r"""
# ---- Input / output ---------------------------------------------------------
# Set AUDIO_PATH to a .wav/.mp3/.flac file, or leave None to synthesize the demo track (section 4).
# The environment variables let you render another song without editing the notebook, e.g.
#   AUDIO_PATH="song.mp3" CLIP_START=28 jupyter nbconvert --execute ...
AUDIO_PATH    = os.environ.get("AUDIO_PATH") or None
CLIP_START    = float(os.environ.get("CLIP_START", 0.0))   # seconds into the file where the clip starts
CLIP_DURATION = 30.0                                       # seconds

# Visual style: "psychedelic" = endless kaleidoscope zoom, morph transitions, colour cycling (section 13)
#               "modes"       = a different motion mode per shape family (spin, sway, shatter, flow, ...)
VISUAL_STYLE  = os.environ.get("VISUAL_STYLE", "psychedelic")

# Each song gets its own output folder (the demo track writes to outputs/ directly).
def slugify(text):
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")

RUN_NAME   = "demo" if AUDIO_PATH is None else slugify(Path(AUDIO_PATH).stem)
OUTPUT_DIR = Path("outputs") if AUDIO_PATH is None else Path("outputs") / RUN_NAME

# ---- Audio analysis -----------------------------------------------------------
SR         = 22050            # sample rate (Hz)
N_FFT      = 2048             # STFT window size (samples)
HOP_LENGTH = 512              # STFT hop (samples) -> one feature frame every 23.2 ms; used everywhere
SMOOTH_SEC = 0.25             # rolling-mean window for continuous features (seconds)
NORM_PERCENTILES = (5, 95)    # robust normalization: clip to these percentiles, then scale to [0, 1]

# Frequency bands (Hz). These are design choices, not psychoacoustic boundaries.
# The high band is capped below the Nyquist frequency (SR / 2) in section 5.
BANDS = {"bass": (20, 250), "mid": (250, 2000), "high": (2000, 8000)}

PULSE_DECAY = 8.0             # k in pulse(t) = exp(-k * time_since_event); 8 -> half-life ~ 87 ms

# ---- Instrument stems (section 5.1) ----------------------------------------------------
DEMUCS_MODEL    = "htdemucs"  # pretrained source-separation model: drums / bass / other / vocals
STEM_PRESENCE   = 0.15        # a stem counts as "present" if its loud level is >= 15% of the full mix

# ---- Vibe (section 9.1) ------------------------------------------------------------------
CLAP_MODEL      = "laion/clap-htsat-unfused"
VIBE_WEIGHTS    = {"acoustic": 0.5, "clap": 0.5}   # how the two audio opinions are combined
LYRICS_ANALYSIS = True        # transcribe the vocal stem + sentiment (only if the song has vocals)
WHISPER_MODEL   = "openai/whisper-small"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"   # positive / neutral / negative
BITTERSWEET_NEGATIVITY = 0.50 # upbeat audio + lyrics mostly negative (top class and >= 50%) -> "bittersweet"
VIBE_OVERRIDE   = os.environ.get("VIBE_OVERRIDE") or None   # e.g. "dark" to force a vibe; None = detect it
INSTRUMENT_ON, INSTRUMENT_OFF = 0.12, 0.06   # stem level (fraction of the mix) to switch an instrument on / off
INSTRUMENT_HOLD_SEC = 2.0     # an instrument must stay on/off at least this long (ignores pauses between phrases)
INSTRUMENT_SMOOTH_SEC = 1.0   # stem levels are averaged over this window before switching
FOLD_MORPH_SEC      = 0.75    # how long the kaleidoscope takes to change its number of mirror segments
DRIVE_THRESHOLD = 0.35        # drum share of stem energy above which the layout becomes "tunnel rings" (section 9.2)

# ---- Sections -------------------------------------------------------------------
N_SECTIONS      = 4           # target number of analysis sections (3-4)
MIN_SECTION_SEC = 4.5         # minimum section length; also the minimum spacing between boundaries
ACTIVITY_WEIGHTS = {"rms": 0.50, "onset": 0.30, "bass": 0.20}
ACTIVITY_SMOOTH_SEC = 1.0     # extra smoothing before looking for change points

# ---- Dramatic beat changes (section 9.3) ----------------------------------------------------
DRAMATIC_WINDOW_SEC = 3.0     # compare this many seconds before vs after each section boundary
DRAMATIC_WEIGHTS    = {"drums": 1.0, "drum_hits": 1.0, "bass": 1.0, "tempo": 0.5}   # tempo is noisy -> half weight
DRAMATIC_THRESHOLD  = 0.50    # largest weighted change at or above this -> completely new image and shape
DRAMATIC_CROSSFADE  = 1.5 if VISUAL_STYLE == "psychedelic" else 0.25   # seconds: morph (psychedelic) / near-cut
DRAMATIC_FLASH      = 0.0 if VISUAL_STYLE == "psychedelic" else 0.35   # extra flash at a dramatic change (no flash = no cut)

# ---- Image generation ---------------------------------------------------------------
MODEL_ID        = "stabilityai/sd-turbo"
IMAGE_SIZE      = 512         # SD-Turbo's native resolution
GEN_STEPS       = 2           # SD-Turbo is designed for 1-4 steps
RANDOM_SEED     = 42          # fixed seed for the particle field; image seeds come from the song (section 9.2)
IMG2IMG_STRENGTH = 0.6        # sections 2+ start from the previous section's image (0 = copy, 1 = ignore it)
IMG2IMG_STEPS    = 4          # img2img runs int(steps x strength) = 2 denoising steps
FORCE_REGENERATE = False      # True = ignore cached images and generate again

# ---- Video ----------------------------------------------------------------------------
FPS               = 24
OUTPUT_SIZE       = 512       # square video, pixels
CROSSFADE_SECONDS = 2.5 if VISUAL_STYLE == "psychedelic" else 1.0   # length of a normal transition (s)
PREVIEW_SECONDS   = 8.0       # length of the quick preview render

# ---- Visual mapping coefficients (section 12) ------------------------------------------
# Image parts are defined by radius from the centre (0 = centre, 1 = edge of the frame).
VIS = {
    "core_radius": 0.24, "inner_radius": 0.58, "mask_softness": 0.08,   # core | inner star | outer star
    "zoom_base": 1.00, "zoom_bass": 0.04,                  # whole-image zoom
    "core_scale_bass": 0.22, "core_scale_beat": 0.10,      # core grows with the bass stem and on beats
    "core_rot_base": 4.0,                                  # core rotation, deg/s
    "inner_rot_base": 8.0, "inner_rot_lead": 80.0,         # inner star rotation (clockwise), deg/s
    "swirl_lead": 1.6,                                     # extra twist (radians) across the inner star
    "outer_rot_base": 5.0, "outer_rot_other": 45.0,        # outer star rotation (counter-clockwise), deg/s
    "outer_scale_beat": 0.05,                              # outer star pulses on beats
    "ripple_high": 0.012, "ripple_waves": 7, "ripple_speed": 2.5,   # fine ripple from the high band
    "lines_base": 0.05, "lines_drums": 0.55, "lines_stretch": 0.06, # line art flashes/stretches on drum hits
    "lights_base": 0.15, "lights_drums": 0.65, "lights_rms": 0.10, "light_hue_step": 0.13,  # light spots recolour
    "particles_min": 30, "particles_range": 250,           # number of visible particles (high band)
    "brightness_base": 0.70, "brightness_rms": 0.25,       # overall gain
    "saturation_base": 0.80, "saturation_rms": 0.50,
    "sharpness_base": 0.90, "sharpness_centroid": 0.30,    # >1 sharpens, <1 softens
    "flash_onset": 0.12,                                   # max additive white flash
    # Motion modes (section 12): each shape family moves in its own way
    "drift_amp": 0.035, "drift_speed_base": 0.05, "drift_speed_lead": 0.15,  # slow camera drift (all modes)
    "sway_deg": 28.0, "sway_speed_base": 0.25, "sway_speed_lead": 0.5,   # sway: rock back and forth (cycles/s)
    "zoom_push": 0.70,                                     # zoom: fly into the image over the section
    "shock_speed": 1.6, "shock_amp": 0.06, "shock_width": 0.07,          # shockwave: ring wave from each beat
    "shatter_tiles": 6, "shatter_amp": 0.07,               # shatter: tiles jump apart on drum hits
    "flow_amp_base": 0.010, "flow_amp_lead": 0.030, "flow_speed_base": 0.3, "flow_speed_lead": 1.2,  # flow: liquid
    # Psychedelic style (section 13)
    "zoom_speed_base": 0.05, "zoom_speed_rms": 0.15, "zoom_speed_beat": 0.90,  # endless zoom: surges forward on beats
    "kaleido_rot_base": 3.0, "kaleido_rot_mid": 12.0,      # wedge rotation, deg/s
    "soft_attack_sec": 0.12,                               # psychedelic: beat/drum/onset pulses swell in over ~2 frames
    "psy_flash_scale": 0.4,                                # psychedelic: onset flashes at 40% strength
    "beat_zoom_amp": 0.10,                                 # zoom punch: +10% zoom on every beat, easing back out
    "instrument_line_gain": 0.9,                           # brightness of the instrument-coloured line art
    "instrument_entry_glow": 0.45,                         # bloom of an instrument's ring when it enters
    "hue_speed_base": 0.02, "hue_speed_rms": 0.10, "hue_jump_drum": 0.03,     # colour cycling, turns/s and per hit
    "morph_swirl": 2.5,                                    # radians of swirl at the middle of a morph
    "beat_saturation": 0.35,                               # extra saturation on each beat
    "psy_saturation": 1.45, "psy_contrast": 1.30,          # vivid neon look (psychedelic style only)
}

# ---- Baseline (section 15): fixed rules that never read the audio ------------------------
BASELINE_PULSE_PERIOD  = 0.70   # seconds between preset pulses (deliberately not the song's tempo)
BASELINE_FLASH_PERIOD  = 2.80   # seconds between preset flashes
BASELINE_BRIGHT_PERIOD = 10.0   # seconds per slow brightness cycle
BASELINE_LEVEL         = 0.5    # fixed "medium" value used in place of every continuous audio control

RENDER_SHIFTED_CHECK   = False  # optional robustness check: audio controls circularly shifted in time
BASELINE_SHIFT_SECONDS = 6.0

# ---- Derived paths --------------------------------------------------------------------------
IMAGE_DIR = OUTPUT_DIR / "images"
FIG_DIR   = OUTPUT_DIR / "figures"
STEMS_DIR = OUTPUT_DIR / "stems"
for d in (OUTPUT_DIR, IMAGE_DIR, FIG_DIR, STEMS_DIR):
    d.mkdir(parents=True, exist_ok=True)
""")

# =============================================================================
# 4. Load audio
# =============================================================================
md(r"""
## 4. Load and preview audio

If `AUDIO_PATH` is set, the clip is loaded, resampled to `SR`, converted to mono and trimmed to `CLIP_DURATION`.

If it is `None`, a **30-second electronic demo track** is synthesized with NumPy so the notebook runs anywhere without a music file
(and without copyright issues on GitHub). The demo has four parts that change in clearly different ways, which makes it easy to check
whether the section detector and the visual mappings behave sensibly:

| Time | Content | What it should look like in the features |
|---|---|---|
| 0–8 s | soft sustained chord pad | low energy, few onsets |
| 8–16 s | + kick drum and bass line | bass-dominant, regular beats |
| 16–24 s | + hi-hats, clap and arpeggio | high energy, bright, many onsets |
| 24–30 s | breakdown: kick and bass drop out | less bass, high-frequency detail remains |
""")

code(r"""
def synthesize_demo_track(duration=30.0, sr=22050, bpm=120, seed=0):
    # Builds a simple 4-part electronic track from sine waves and filtered noise.
    rng = np.random.default_rng(seed)
    n = int(duration * sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    beat = 60.0 / bpm                        # seconds per beat (0.5 s at 120 BPM)

    def add(start_sec, sig):
        # Add a short sound event starting at start_sec (clipped at the end of the track).
        i = int(start_sec * sr)
        if i < n:
            m = min(len(sig), n - i)
            y[i:i + m] += sig[:m]

    def decay(length_sec, k):
        return np.exp(-k * np.arange(int(length_sec * sr)) / sr)

    # Pad: A-minor chord, present all the time, slightly louder in the breakdown.
    pad = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 261.6, 329.6)) / 3
    pad_gain = np.where(t < 24, 0.12, 0.18)
    y += pad_gain * pad

    for b in np.arange(0, duration, beat):
        beat_idx = int(round(b / beat))
        # Kick + bass: 8-24 s
        if 8 <= b < 24:
            tt = np.arange(int(0.30 * sr)) / sr
            freq = 45 + 75 * np.exp(-tt * 25)               # pitch sweep 120 Hz -> 45 Hz
            add(b, 0.9 * np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-tt * 9))
            bass_t = np.arange(int(0.45 * beat * sr)) / sr  # bass note on the off-beat
            add(b + beat / 2, 0.35 * np.sin(2 * np.pi * 55 * bass_t) * decay(0.45 * beat, 4))
        # Hi-hats, clap, arpeggio: 16-30 s
        if b >= 16:
            for off in (0.0, 0.25, 0.5, 0.75):              # 16th-note hats
                noise = rng.standard_normal(int(0.04 * sr))
                hat = np.diff(noise, prepend=0)              # crude high-pass filter
                add(b + off * beat, 0.08 * hat * decay(0.04, 80))
            if beat_idx % 2 == 1 and b < 24:                 # clap on beats 2 and 4
                add(b, 0.25 * rng.standard_normal(int(0.12 * sr)) * decay(0.12, 30))
            notes = (880.0, 1046.5, 1318.5, 1760.0)          # A5 C6 E6 A6
            for k, off in enumerate((0.0, 0.25, 0.5, 0.75)):
                at = np.arange(int(0.11 * sr)) / sr
                f = notes[(beat_idx * 4 + k) % 4]
                add(b + off * beat, 0.10 * np.sign(np.sin(2 * np.pi * f * at)) * decay(0.11, 30))

    y = y / (np.max(np.abs(y)) + 1e-9) * 0.9                  # normalize peak to 0.9
    return y.astype(np.float32)


if AUDIO_PATH is None:
    y = synthesize_demo_track(CLIP_DURATION, SR)
    AUDIO_SOURCE = "synthesized demo track (120 BPM)"
else:
    y, _ = librosa.load(AUDIO_PATH, sr=SR, mono=True, offset=CLIP_START, duration=CLIP_DURATION)
    AUDIO_SOURCE = str(AUDIO_PATH)

DURATION = len(y) / SR
AUDIO_WAV = OUTPUT_DIR / "audio_clip.wav"
sf.write(AUDIO_WAV, y, SR)                       # exact audio that will be muxed into the videos

print(f"Source: {AUDIO_SOURCE}" + (f" (clip {CLIP_START:.1f}-{CLIP_START + DURATION:.1f} s)" if AUDIO_PATH else ""))
print(f"Outputs: {OUTPUT_DIR}/")
print(f"Samples: {len(y):,} | sample rate: {SR} Hz | duration: {DURATION:.2f} s")
assert abs(DURATION - CLIP_DURATION) < 0.5, "Clip is shorter than CLIP_DURATION; check AUDIO_PATH / CLIP_START"
""")

code(r"""
t_audio = np.arange(len(y)) / SR
fig, ax = plt.subplots(figsize=(12, 2.6))
ax.plot(t_audio, y, color=COLORS["rms"], lw=0.4)
ax.set(title="Waveform", xlabel="time (s)", ylabel="amplitude", xlim=(0, DURATION))
plt.tight_layout(); plt.savefig(FIG_DIR / "waveform.png"); plt.show()

display(Audio(y, rate=SR))
""")

# =============================================================================
# 5. Features
# =============================================================================
md(r"""
## 5. Extract audio features

All features use the same `HOP_LENGTH`, so every feature has one value per STFT frame (every 23.2 ms) and shares one time axis.

| Feature | LibROSA function | What it measures | Note |
|---|---|---|---|
| Onset strength | `onset.onset_strength` | spectral flux: how quickly the spectrum changes | high at drum hits / note attacks |
| Beats + tempo | `beat.beat_track` | estimated beat times | an estimate, not ground truth |
| RMS energy | `feature.rms` | short-time energy / amplitude | a proxy, **not** perceived loudness |
| Spectral centroid | `feature.spectral_centroid` | "centre of mass" of the spectrum | a **brightness** proxy, **not** pitch |
| Band energy | STFT power averaged over frequency bins | energy in low / mid / high bands | bands defined in the config |

Band energy is computed as `mean(|STFT|² over the bins in the band)`, then converted to decibels so that quiet and loud passages both
show useful variation (a design choice; normalization in section 6 then maps it to [0, 1]).
""")

code(r"""
# Short-time Fourier transform -> power spectrogram (frequency bins x frames)
S_power = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2
freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)          # centre frequency of each STFT bin
n_feat_frames = S_power.shape[1]
feat_times = librosa.frames_to_time(np.arange(n_feat_frames), sr=SR, hop_length=HOP_LENGTH)

# Onset strength (spectral flux) and onset events
onset_env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP_LENGTH)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR, hop_length=HOP_LENGTH)
onset_times = librosa.frames_to_time(onset_frames, sr=SR, hop_length=HOP_LENGTH)

# Beat tracking (tempo may come back as a 1-element array)
tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR, hop_length=HOP_LENGTH)
tempo = float(np.atleast_1d(tempo)[0])
beat_times = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP_LENGTH)

# RMS energy and spectral centroid
rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
centroid = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]

# Band energies: average power over the bins inside each band, in dB
nyquist = SR / 2
band_db = {}
for name, (lo, hi) in BANDS.items():
    hi = min(hi, nyquist - 1)                                  # cap the band below Nyquist
    mask = (freqs >= lo) & (freqs < hi)
    band_db[name] = librosa.power_to_db(S_power[mask].mean(axis=0), ref=np.max(S_power))
    print(f"{name:>4} band: {lo:>5}-{hi:>6.0f} Hz  ({mask.sum()} STFT bins)")

raw = pd.DataFrame({"time_sec": feat_times, "rms": rms, "centroid": centroid, "onset": onset_env, **band_db})
assert raw.notna().all().all() and np.isfinite(raw.values).all(), "features contain NaN/inf"

print(f"\nFeature frames: {n_feat_frames} (one every {HOP_LENGTH / SR * 1000:.1f} ms)")
print(f"Estimated tempo: {tempo:.1f} BPM | beats: {len(beat_times)} | onsets: {len(onset_times)}")
raw.describe().T[["mean", "min", "max"]].round(3)
""")

code(r"""
fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
axes[0].plot(t_audio, y, color=COLORS["rms"], lw=0.3)
axes[0].vlines(beat_times, -1, 1, color=COLORS["beat"], lw=0.8, ls="--", label="detected beats")
axes[0].set(ylabel="amplitude", title="Waveform with detected beats"); axes[0].legend(loc="upper left")

axes[1].plot(feat_times, onset_env, color=COLORS["onset"], lw=0.8, label="onset strength")
axes[1].plot(onset_times, onset_env[onset_frames], "o", ms=3, color=COLORS["onset"], label="detected onsets")
axes[1].set(ylabel="onset strength", title="Onset strength (spectral flux)"); axes[1].legend(loc="upper left")

axes[2].plot(feat_times, rms, color=COLORS["rms"], lw=0.8)
axes[2].set(ylabel="RMS", title="RMS energy (amplitude proxy)")

axes[3].plot(feat_times, centroid, color=COLORS["centroid"], lw=0.8)
axes[3].set(ylabel="Hz", title="Spectral centroid (brightness proxy, not pitch)")

for name in BANDS:
    axes[4].plot(feat_times, band_db[name], color=COLORS[name], lw=0.8, label=name)
axes[4].set(ylabel="dB", xlabel="time (s)", title="Band energy"); axes[4].legend(loc="lower left", ncol=3)
axes[4].set_xlim(0, DURATION)
plt.tight_layout(); plt.savefig(FIG_DIR / "features_raw.png"); plt.show()
""")

# =============================================================================
# 5.1 Instrument stems
# =============================================================================
md(r"""
### 5.1 Separate the instruments (Demucs stems)

Frequency bands mix every instrument together: a bass guitar, a kick drum and the low end of a vocal all land in the "bass" band.
To react to **instruments**, the clip is separated into four stems with the pretrained **[Demucs](https://github.com/adefossez/demucs)**
model (`htdemucs`): **drums, bass, other** (synths, guitars, keys) and **vocals**.

Each stem then gets its own RMS envelope, and the drum stem gets its own onset (hit) detector. Separation is imperfect: some sound
leaks between stems. A stem whose loud level stays below `STEM_PRESENCE` × the full mix is treated as **absent** (e.g. no vocals in an
instrumental) so that leakage noise is not amplified by normalization.

Stems are cached in `outputs/.../stems/`. If Demucs is unavailable, a clearly labelled **proxy** is used instead
(harmonic/percussive separation: percussive ≈ drums, low harmonic ≈ bass, other harmonic ≈ other, no vocals).
""")

code(r"""
STEM_NAMES = ["drums", "bass", "other", "vocals"]
stem_key = {"source": AUDIO_SOURCE, "start": CLIP_START, "duration": round(DURATION, 3), "model": DEMUCS_MODEL}
key_file = STEMS_DIR / "source.json"

def load_cached_stems():
    if key_file.exists() and json.loads(key_file.read_text()) == stem_key:
        if all((STEMS_DIR / f"{n}.wav").exists() for n in STEM_NAMES):
            return {n: librosa.load(STEMS_DIR / f"{n}.wav", sr=SR)[0] for n in STEM_NAMES}
    return None

def demucs_stems(y):
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    model = get_model(DEMUCS_MODEL).eval()
    y_model = librosa.resample(y, orig_sr=SR, target_sr=model.samplerate)
    wav = torch.tensor(np.stack([y_model, y_model]), dtype=torch.float32)[None]   # (batch, stereo, samples)
    device = "cuda" if DEVICE == "cuda" else "cpu"       # MPS lacks an operation Demucs needs
    with torch.no_grad():
        out = apply_model(model, wav, device=device, split=True, overlap=0.25, progress=False)[0]
    return {name: librosa.resample(out[i].mean(0).numpy(), orig_sr=model.samplerate, target_sr=SR)
            for i, name in enumerate(model.sources)}

def proxy_stems(y):
    # Fallback without Demucs: harmonic/percussive separation + a low-pass split of the harmonic part.
    harmonic, percussive = librosa.effects.hpss(y)
    H = librosa.stft(harmonic, n_fft=N_FFT, hop_length=HOP_LENGTH)
    low = (freqs < BANDS["bass"][1])[:, None]
    return {"drums": percussive,
            "bass": librosa.istft(H * low, hop_length=HOP_LENGTH, length=len(y)),
            "other": librosa.istft(H * ~low, hop_length=HOP_LENGTH, length=len(y)),
            "vocals": np.zeros_like(y)}

stems = load_cached_stems()
if stems is not None:
    STEM_METHOD = f"Demucs {DEMUCS_MODEL} (cached)"
else:
    try:
        stems = demucs_stems(y)
        STEM_METHOD = f"Demucs {DEMUCS_MODEL}"
    except Exception as e:
        print(f"Demucs unavailable ({type(e).__name__}: {e}); using PROXY stems")
        stems = proxy_stems(y)
        STEM_METHOD = "proxy (HPSS)"
    for name in STEM_NAMES:
        sf.write(STEMS_DIR / f"{name}.wav", stems[name], SR)
    key_file.write_text(json.dumps(stem_key))
stems = {n: librosa.util.fix_length(stems[n], size=len(y)) for n in STEM_NAMES}

# Per-stem RMS on the same frames as every other feature
stem_rms = {n: librosa.feature.rms(y=stems[n], frame_length=N_FFT, hop_length=HOP_LENGTH)[0][:n_feat_frames]
            for n in STEM_NAMES}
mix_level = np.percentile(rms, 95)
stem_present = {n: bool(np.percentile(stem_rms[n], 95) >= STEM_PRESENCE * mix_level) for n in STEM_NAMES}

# Drum hits: onset detection on the drum stem only
drum_env = librosa.onset.onset_strength(y=stems["drums"], sr=SR, hop_length=HOP_LENGTH)
drum_frames = librosa.onset.onset_detect(onset_envelope=drum_env, sr=SR, hop_length=HOP_LENGTH)
drum_times = librosa.frames_to_time(drum_frames, sr=SR, hop_length=HOP_LENGTH)
drum_strength = drum_env[drum_frames] / (np.percentile(drum_env[drum_frames], 95) + 1e-9) if len(drum_frames) else []
drum_strength = np.clip(drum_strength, 0, 1)

print(f"Stem method: {STEM_METHOD}")
for n in STEM_NAMES:
    print(f"  {n:<7} loud level {np.percentile(stem_rms[n], 95) / mix_level:5.0%} of mix -> "
          f"{'present' if stem_present[n] else 'ABSENT (treated as silent)'}")
print(f"Drum hits detected: {len(drum_times)}")
""")

code(r"""
fig, axes = plt.subplots(4, 1, figsize=(12, 7.5), sharex=True)
for ax, n in zip(axes, STEM_NAMES):
    color = COLORS["bass_stem" if n == "bass" else n]
    ax.plot(feat_times, stem_rms[n], color=color, lw=0.9)
    if n == "drums":
        ax.vlines(drum_times, 0, stem_rms[n].max(), color=COLORS["beat"], lw=0.5, ls=":", label="drum hits")
        ax.legend(loc="upper left")
    ax.set(ylabel="RMS", title=f"{n} stem" + ("" if stem_present[n] else " (absent)"))
axes[-1].set(xlabel="time (s)", xlim=(0, DURATION))
plt.tight_layout(); plt.savefig(FIG_DIR / "stems.png"); plt.show()
display(Audio(stems["drums"], rate=SR))
""")

# =============================================================================
# 6. Smooth & normalize
# =============================================================================
md(r"""
## 6. Smooth and normalize features

Raw features jump around from frame to frame, which would make the animation jittery. Two steps fix this:

1. **Smoothing:** a rolling mean over `SMOOTH_SEC` (0.25 s ≈ 11 frames) for the continuous features (RMS, centroid, bands).
   Onset strength is *not* smoothed, because its job is to mark sharp events.
2. **Robust normalization:** clip each feature to its 5th–95th percentile, then min-max scale to [0, 1].
   Clipping stops one extreme spike from squashing the rest of the signal.

Each feature is normalized independently, so `bass = 1` means "as bassy as this clip gets", not an absolute level.
""")

code(r"""
def robust_normalize(x, lo_pct=NORM_PERCENTILES[0], hi_pct=NORM_PERCENTILES[1]):
    # Clip to percentiles, then scale to [0, 1].
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)

smooth_frames = max(1, int(round(SMOOTH_SEC * SR / HOP_LENGTH)))
print(f"Smoothing window: {smooth_frames} feature frames ({smooth_frames * HOP_LENGTH / SR:.2f} s)")

feat = pd.DataFrame({"time_sec": feat_times})
for name in ["rms", "centroid", "bass", "mid", "high"]:
    smoothed = uniform_filter1d(raw[name].values.astype(float), size=smooth_frames, mode="nearest")
    feat[name] = robust_normalize(smoothed)
feat["onset"] = robust_normalize(raw["onset"].values)       # events: normalize only

# Instrument stems: same smoothing + normalization; absent stems stay at zero
for n in STEM_NAMES:
    col = "bass_stem" if n == "bass" else n
    smoothed = uniform_filter1d(stem_rms[n].astype(float), size=smooth_frames, mode="nearest")
    feat[col] = robust_normalize(smoothed) if stem_present[n] else 0.0

assert feat.drop(columns="time_sec").apply(lambda c: c.between(0, 1).all()).all()
feat.describe().T[["mean", "std", "min", "max"]].round(3)
""")

code(r"""
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
axes[0].plot(feat_times, robust_normalize(raw["rms"].values), color="#c3c2b7", lw=0.8, label="raw (normalized only)")
axes[0].plot(feat_times, feat["rms"], color=COLORS["rms"], lw=1.6, label="smoothed + normalized")
axes[0].set(title="RMS before and after smoothing", ylabel="RMS (0-1)"); axes[0].legend(loc="upper left")
for name in ["bass", "mid", "high", "centroid"]:
    axes[1].plot(feat_times, feat[name], color=COLORS[name], lw=1.2, label=name)
axes[1].set(title="Smoothed, normalized continuous features", xlabel="time (s)", ylabel="value (0-1)", xlim=(0, DURATION))
axes[1].legend(loc="upper left", ncol=4)
plt.tight_layout(); plt.savefig(FIG_DIR / "features_normalized.png"); plt.show()
""")

# =============================================================================
# 7. Control timeline
# =============================================================================
md(r"""
## 7. Build the frame-level control timeline

The video runs at `FPS` = 24 frames per second, but the features exist every 23.2 ms. `np.interp` resamples every feature onto the
**video frame timestamps** (`frame / FPS`), producing one row per video frame. This DataFrame is the **single source of truth** for
the renderer, which never touches the audio directly.

Beats and onsets are single moments in time. If they affected only one frame they would be invisible, so each event becomes a
smooth envelope with a sharp attack and a fast decay:

$$\text{pulse}(t) = e^{-k\,(t - t_{\text{last event}})}, \qquad k = \texttt{PULSE\_DECAY}$$

For onsets the envelope is scaled by the onset's normalized strength, so stronger attacks flash brighter.
Onset strength is spiky (peaks only 1–2 feature frames wide), so it is max-pooled over one video-frame interval before
interpolation; otherwise sampling at 24 fps would miss many onsets. `section_id` is filled in after section detection (section 8).
""")

code(r"""
N_FRAMES = int(round(DURATION * FPS))
frame_times = np.arange(N_FRAMES) / FPS

def event_envelope(event_times, times, k=PULSE_DECAY, weights=None):
    # exp(-k * time since the most recent event); 0 before the first event.
    event_times = np.asarray(event_times, float)
    if len(event_times) == 0:
        return np.zeros_like(times)
    weights = np.ones_like(event_times) if weights is None else np.asarray(weights, float)
    idx = np.searchsorted(event_times, times, side="right") - 1   # most recent event at or before t
    dt = times - event_times[np.clip(idx, 0, None)]
    return np.where(idx >= 0, weights[np.clip(idx, 0, None)] * np.exp(-k * dt), 0.0)

timeline = pd.DataFrame({"frame": np.arange(N_FRAMES), "time_sec": frame_times})
for name in ["rms", "bass", "mid", "high", "centroid"]:
    timeline[name] = np.interp(frame_times, feat_times, feat[name])
# Onset spikes are only 1-2 feature frames wide, so plain interpolation at 24 fps would skip many of them.
# Take the local maximum over one video-frame interval first, then interpolate.
per_video_frame = int(np.ceil(SR / HOP_LENGTH / FPS)) + 1
timeline["onset"] = np.interp(frame_times, feat_times, maximum_filter1d(feat["onset"].values, size=per_video_frame))
timeline["beat_pulse"] = event_envelope(beat_times, frame_times)
timeline["onset_pulse"] = event_envelope(onset_times, frame_times, weights=feat["onset"].values[onset_frames])

# Instrument controls
for col in ["drums", "bass_stem", "other", "vocals"]:
    timeline[col] = np.interp(frame_times, feat_times, feat[col])
# "lead" = the melodic voice that twists the inner star: vocals if the song has them, otherwise the mid band
LEAD_SOURCE = "vocals" if stem_present["vocals"] else "mid band"
timeline["lead"] = timeline["vocals"] if stem_present["vocals"] else timeline["mid"]
timeline["drum_pulse"] = event_envelope(drum_times, frame_times, weights=drum_strength)
timeline["drum_hits"] = np.searchsorted(drum_times, frame_times, side="right")   # running count (light colours)
timeline["drum_unit"] = event_envelope(drum_times, frame_times)   # unweighted decay after each hit (smooth colour steps)
timeline["section_id"] = 0          # placeholder, set in section 8

# Acceptance checks
assert len(timeline) == N_FRAMES == round(DURATION * FPS)
assert not timeline.isna().any().any(), "NaNs in timeline"
controls = ["rms", "bass", "mid", "high", "centroid", "onset", "beat_pulse", "onset_pulse",
            "drums", "bass_stem", "other", "vocals", "lead", "drum_pulse"]
assert timeline[controls].apply(lambda c: c.between(0, 1).all()).all(), "controls out of [0, 1]"
print(f"Timeline: {N_FRAMES} frames = {DURATION:.2f} s x {FPS} fps, all controls within [0, 1], no NaNs")
print(f"Lead (inner-star twist) follows: {LEAD_SOURCE}")
timeline.head(8).round(3)
""")

code(r"""
fig, ax = plt.subplots(figsize=(12, 2.8))
seg = timeline[(timeline.time_sec >= 8) & (timeline.time_sec < 14)]
ax.plot(seg.time_sec, seg.beat_pulse, color=COLORS["beat"], lw=1.5, label="beat pulse envelope")
ax.plot(seg.time_sec, seg.drum_pulse, color=COLORS["drums"], lw=1.2, label="drum-hit pulse")
ax.plot(seg.time_sec, seg.bass_stem, color=COLORS["bass"], lw=1.5, label="bass stem (smoothed)")
ax.vlines(beat_times[(beat_times >= 8) & (beat_times < 14)], 0, 1, color=COLORS["beat"], lw=0.6, ls=":")
ax.set(title="Control timeline close-up (8-14 s): beats (dotted) and drum hits become decaying pulses",
       xlabel="time (s)", ylabel="value (0-1)")
ax.legend(loc="upper right")
plt.tight_layout(); plt.savefig(FIG_DIR / "timeline_closeup.png"); plt.show()
""")

# =============================================================================
# 7.1 Instrument activity
# =============================================================================
md(r"""
### 7.1 When does each instrument play? (entrances and exits)

To show **changes of instrument inside the design**, each stem is turned into an on/off state per video frame:

* level = stem RMS averaged over `INSTRUMENT_SMOOTH_SEC` (1 s), as a fraction of the full mix's loud level;
* an instrument switches **on** above `INSTRUMENT_ON` (12 %) and **off** below `INSTRUMENT_OFF` (6 %) (hysteresis, so it does not
  flicker around one threshold), and must hold each state for `INSTRUMENT_HOLD_SEC` (2 s), so short pauses between phrases
  are not counted as the instrument leaving.

Every switch is an **instrument change event** (e.g. "vocals enter at 4.2 s"). In the renderer (section 12):

| Instrument | Part of the design | Colour | What you see |
|---|---|---|---|
| bass | core | orange | core lines glow orange as loud as the bass; core blooms when the bass enters |
| vocals (lead) | inner star | magenta | inner-star lines glow with the vocals (mid band if there are no vocals) |
| other (synths, guitars, keys) | outer ring | cyan | outer lines glow with the other instruments |
| drums | light spots | (colour cycling) | lights flare on hits and when the drums enter |

and the **number of kaleidoscope mirror segments** follows how many instruments are playing
(`N_POINTS + 2 × (active − 2)`, between 4 and 16), morphing over `FOLD_MORPH_SEC` whenever an instrument enters or leaves.
""")

code(r"""
INSTRUMENT_COLORS = {"bass": (1.00, 0.55, 0.15), "vocals": (1.00, 0.30, 0.75),
                     "other": (0.30, 0.90, 1.00), "drums": (1.00, 0.95, 0.70)}

def on_off(level, times, on=INSTRUMENT_ON, off=INSTRUMENT_OFF, hold=INSTRUMENT_HOLD_SEC):
    # Hysteresis switch with a minimum hold time.
    state, active, last = np.zeros(len(level), int), False, -np.inf
    for i, (x, t) in enumerate(zip(level, times)):
        if not active and x >= on and t - last >= hold:
            active, last = True, t
        elif active and x < off and t - last >= hold:
            active, last = False, t
        state[i] = active
    return state

events = []
for n in STEM_NAMES:
    win = max(1, int(round(INSTRUMENT_SMOOTH_SEC * SR / HOP_LENGTH)))
    smoothed = uniform_filter1d(stem_rms[n].astype(float), size=win, mode="nearest") / mix_level
    level = np.interp(frame_times, feat_times, smoothed)
    timeline[f"active_{n}"] = on_off(level, frame_times) if stem_present[n] else 0
    changes = np.flatnonzero(np.diff(timeline[f"active_{n}"].values)) + 1
    events += [{"time_sec": round(frame_times[i], 2), "instrument": n,
                "change": "enters" if timeline[f"active_{n}"].values[i] else "leaves"} for i in changes]
    entries = [frame_times[i] for i in changes if timeline[f"active_{n}"].values[i]]
    if timeline[f"active_{n}"].values[0]:
        entries = [0.0] + entries
    timeline[f"enter_{n}"] = event_envelope(entries, frame_times, k=3.0)   # ~0.3 s bloom after an entrance
timeline["active_count"] = timeline[[f"active_{n}" for n in STEM_NAMES]].sum(axis=1)
instrument_events = pd.DataFrame(events, columns=["time_sec", "instrument", "change"]).sort_values("time_sec")
instrument_events.to_csv(OUTPUT_DIR / "instrument_events.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 2.6))
for k, n in enumerate(STEM_NAMES):
    on = timeline[f"active_{n}"].values.astype(bool)
    starts = np.flatnonzero(np.diff(np.r_[0, on.astype(int)]) == 1)
    ends = np.flatnonzero(np.diff(np.r_[on.astype(int), 0]) == -1)
    ax.broken_barh([(frame_times[a], frame_times[e] - frame_times[a] + 1 / FPS) for a, e in zip(starts, ends)],
                   (k - 0.35, 0.7), color=INSTRUMENT_COLORS[n])
ax.set_yticks(range(len(STEM_NAMES)), STEM_NAMES)
ax.set(xlim=(0, DURATION), xlabel="time (s)", title="Instrument activity (bars = playing); every bar edge is an instrument change")
ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(FIG_DIR / "instrument_activity.png"); plt.show()
instrument_events if len(instrument_events) else print("No instrument changes in this clip")
""")

# =============================================================================
# 8. Sections
# =============================================================================
md(r"""
## 8. Detect analysis sections

The generative model needs a reason to make several images, so the clip is split into 3–4 **analysis sections** (change regions).
They are labelled neutrally (Section 1, 2, …); the detector does **not** claim to find verses, choruses or drops.

**Method (simple and interpretable):**
1. `activity = 0.50·rms + 0.30·onset + 0.20·bass` (normalized features)
2. smooth `activity` over `ACTIVITY_SMOOTH_SEC` (1 s)
3. take the absolute first difference: large where activity changes quickly
4. find peaks at least `MIN_SECTION_SEC` apart (and that far from the start/end)
5. keep the `N_SECTIONS − 1` most prominent peaks as boundaries
6. if fewer than 2 boundaries are found, **fall back to equal windows**
""")

code(r"""
w = ACTIVITY_WEIGHTS
activity = w["rms"] * feat["rms"] + w["onset"] * feat["onset"] + w["bass"] * feat["bass"]
act_frames = max(1, int(round(ACTIVITY_SMOOTH_SEC * SR / HOP_LENGTH)))
activity_smooth = uniform_filter1d(activity.values, size=act_frames, mode="nearest")
change = np.abs(np.diff(activity_smooth, prepend=activity_smooth[0]))

min_gap = int(MIN_SECTION_SEC * SR / HOP_LENGTH)            # in feature frames
peaks, props = scipy.signal.find_peaks(change, distance=min_gap, prominence=0)
edge_ok = (feat_times[peaks] >= MIN_SECTION_SEC) & (feat_times[peaks] <= DURATION - MIN_SECTION_SEC)
peaks, prominences = peaks[edge_ok], props["prominences"][edge_ok]

# Keep the most prominent peaks, then make sure every section is long enough
best = np.sort(peaks[np.argsort(prominences)[::-1][:N_SECTIONS - 1]])
boundaries = feat_times[best]
edges = np.concatenate([[0.0], boundaries, [DURATION]])

if len(boundaries) >= 2 and np.all(np.diff(edges) >= MIN_SECTION_SEC):
    SECTION_METHOD = "automatic change-point detection"
else:
    edges = np.linspace(0, DURATION, N_SECTIONS + 1)
    SECTION_METHOD = "equal-window fallback"

section_edges = edges
timeline["section_id"] = np.clip(np.searchsorted(section_edges, frame_times, side="right"), 1, len(section_edges) - 1)
n_sections = len(section_edges) - 1

print(f"Method: {SECTION_METHOD}")
print("Section boundaries (s):", np.round(section_edges, 2))
print("Section lengths (s):   ", np.round(np.diff(section_edges), 2))
assert 3 <= n_sections <= 4 and np.all(np.diff(section_edges) >= MIN_SECTION_SEC - 1e-6)
""")

code(r"""
fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
axes[0].plot(feat_times, activity, color="#c3c2b7", lw=0.6, label="activity")
axes[0].plot(feat_times, activity_smooth, color=COLORS["rms"], lw=1.8, label="activity (smoothed)")
axes[0].set(title="Activity curve and detected section boundaries", ylabel="activity")
axes[1].plot(feat_times, change, color=COLORS["onset"], lw=1.2, label="|first difference|")
axes[1].plot(feat_times[peaks], change[peaks], "o", ms=5, mfc="none", color=COLORS["onset"], label="candidate peaks")
axes[1].set(ylabel="change", xlabel="time (s)", xlim=(0, DURATION))
for ax in axes:
    for b in section_edges[1:-1]:
        ax.axvline(b, color=COLORS["baseline"], lw=1.5, ls="--")
axes[0].legend(loc="center left"); axes[1].legend(loc="upper right")
for i in range(n_sections):
    axes[0].text((section_edges[i] + section_edges[i + 1]) / 2, axes[0].get_ylim()[1] * 0.92,
                 f"Section {i + 1}", ha="center", color="#52514e")
plt.tight_layout(); plt.savefig(FIG_DIR / "sections.png"); plt.show()
""")

# =============================================================================
# 9. Section summary
# =============================================================================
md(r"""
## 9. Summarize sections

Each section is described by the **mean of its normalized features** plus its onset density (onsets per second).
Onset density is also expressed *relative to the busiest section* (`onset_density_rel`, 0–1) so it can use the same thresholds as the other statistics.
""")

code(r"""
rows = []
for s in range(1, n_sections + 1):
    start, end = section_edges[s - 1], section_edges[s]
    part = timeline[timeline.section_id == s]
    n_onsets = np.sum((onset_times >= start) & (onset_times < end))
    rows.append({
        "section": s, "start": start, "end": end, "length": end - start,
        "rms": part.rms.mean(), "bass": part.bass.mean(), "mid": part.mid.mean(),
        "high": part.high.mean(), "centroid": part.centroid.mean(),
        "onset_density": n_onsets / (end - start),
    })
sections = pd.DataFrame(rows)
sections["onset_density_rel"] = sections.onset_density / sections.onset_density.max()
sections["dominant_band"] = sections[["bass", "mid", "high"]].idxmax(axis=1)

sections.to_csv(OUTPUT_DIR / "sections.csv", index=False)
sections.round(2)
""")

# =============================================================================
# 9.1 Vibe
# =============================================================================
md(r"""
## 9.1 Classify the vibe

Vibe classification is a Music Information Retrieval (MIR) problem. The notebook combines three independent opinions, following the usual
*audio engineer → data scientist → model* pipeline. Vibes: **party, happy, chill, melancholic, dark, epic** (+ **bittersweet**, see C).

**A. Acoustic profile (the audio engineer's lens, fully interpretable).** Seven measurements, each mapped to [0, 1]:

| Feature | Measured as | Why it matters |
|---|---|---|
| `tempo` | beat-tracker BPM: 80 → 0, 125 → 1 | party sits around 120–130 BPM; chill / melancholic around 60–90 |
| `transients` | drum share of the Demucs stems + onset rate | party songs have hard, frequent hits; chill songs soft attacks |
| `compression` | 1 − dynamic range (95th − 10th percentile of frame loudness, 6 dB → 1, 24 dB → 0) | party mixes are heavily compressed; melancholic ones breathe |
| `brightness` | spectral centroid (1–3.5 kHz) and 85 % roll-off (2.5–7.5 kHz) | bright mixes read happy / energetic; rolled-off highs read dark / sad |
| `major` | Krumhansl–Schmuckler key estimate on chroma (drums removed): major vs minor correlation | major ≈ upbeat, minor ≈ tension / sadness |
| `energy` | mean loudness, −30 dBFS → 0, −10 dBFS → 1 | overall intensity |
| `noisiness` | spectral flatness (log scale) | distortion and noise read aggressive / dark |

Each vibe has a hand-written **prototype** (its expected value for every feature, table below). The acoustic score for a vibe is
`exp(−distance² / 2σ²)` between the song's profile and that prototype, normalized over vibes. These are design rules, not learned weights.

**B. Pretrained audio embedding (transfer learning).** CLAP, a model pretrained on hundreds of thousands of audio–text pairs, scores the
clip against a text description of each vibe (zero-shot), averaged over 10-second windows. It also scores instrument / texture tags used
for materials in section 9.2. The per-window scores show how the vibe **changes over time**.

**C. Lyrics (multi-modal).** If the song has vocals, the separated vocal stem is transcribed with Whisper and the text is scored by a
3-class sentiment model (positive / neutral / negative). If the audio says *party / happy* but **negative** is the most likely lyric class
(and ≥ `BITTERSWEET_NEGATIVITY` = 50 %), the song is classified **bittersweet**.

**Final vibe** = `0.5 × acoustic + 0.5 × CLAP`, then the bittersweet rule. `VIBE_OVERRIDE` forces a vibe.

**What is *not* done here:** training or fine-tuning a CNN on mel-spectrograms, an LSTM over time, or a YAMNet/VGGish classifier head.
All of those need a dataset of songs **labelled by vibe**, which this project does not have; CLAP is used as the pretrained-model stand-in.
""")

code(r"""
VIBE_NAMES = ["party", "happy", "chill", "melancholic", "dark", "epic"]

# ---- A. Acoustic profile ------------------------------------------------------------
PITCHES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
chroma_mean = librosa.feature.chroma_cqt(y=y - stems["drums"], sr=SR, hop_length=HOP_LENGTH).mean(axis=1)
best = {mode: max((np.corrcoef(chroma_mean, np.roll(prof, k))[0, 1], k) for k in range(12))
        for mode, prof in (("major", KS_MAJOR), ("minor", KS_MINOR))}
MODE = max(best, key=lambda m: best[m][0])
key_corr, key_idx = best[MODE]
KEY = f"{PITCHES[key_idx]} {MODE}"

def scale(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0, 1))

rms_db = librosa.amplitude_to_db(rms, ref=1.0)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, roll_percent=0.85)[0]
flatness = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
stem_means = {n: float(np.mean(stem_rms[n])) for n in STEM_NAMES if stem_present[n]}
DRUM_SHARE = stem_means.get("drums", 0.0) / (sum(stem_means.values()) or 1.0)
dynamic_range_db = np.percentile(rms_db, 95) - np.percentile(rms_db, 10)

profile = {
    "tempo":       scale(tempo, 80, 125),
    "transients":  0.5 * scale(DRUM_SHARE, 0.0, 0.5) + 0.5 * scale(len(onset_times) / DURATION, 0, 6),
    "compression": 1 - scale(dynamic_range_db, 6, 24),
    "brightness":  0.5 * scale(centroid.mean(), 1000, 3500) + 0.5 * scale(rolloff.mean(), 2500, 7500),
    "major":       scale(best["major"][0] - best["minor"][0], -0.2, 0.2),
    "energy":      scale(rms_db.mean(), -30, -10),
    "noisiness":   scale(np.log10(flatness.mean() + 1e-9), -3, -1),
}
FEATURE_NAMES = list(profile)

# Expected feature values for each vibe (design prototypes, one row per vibe, columns = FEATURE_NAMES)
PROTOTYPES = pd.DataFrame({
    #               tempo transients compression brightness major energy noisiness
    "party":       [0.95, 0.90, 0.90, 0.75, 0.60, 0.90, 0.50],
    "happy":       [0.65, 0.55, 0.60, 0.80, 0.90, 0.60, 0.30],
    "chill":       [0.20, 0.25, 0.40, 0.40, 0.60, 0.30, 0.20],
    "melancholic": [0.20, 0.20, 0.25, 0.25, 0.10, 0.30, 0.20],
    "dark":        [0.60, 0.70, 0.80, 0.30, 0.10, 0.80, 0.80],
    "epic":        [0.50, 0.60, 0.35, 0.50, 0.50, 0.75, 0.40],
}, index=FEATURE_NAMES).T

PROTOTYPE_SIGMA = 0.35
song_vec = np.array([profile[f] for f in FEATURE_NAMES])
dist2 = ((PROTOTYPES.values - song_vec) ** 2).sum(axis=1)
acoustic = np.exp(-dist2 / (2 * PROTOTYPE_SIGMA ** 2))
acoustic_scores = dict(zip(PROTOTYPES.index, acoustic / acoustic.sum()))

# ---- B. CLAP (pretrained audio-text embedding) ---------------------------------------
VIBE_TEXT = {
    "party":       "energetic party dance music",
    "happy":       "happy upbeat cheerful music",
    "chill":       "chill relaxed laid-back music",
    "melancholic": "sad melancholic emotional music",
    "dark":        "dark intense aggressive music",
    "epic":        "epic cinematic powerful music",
}
INSTRUMENTS = {   # instrument / texture tag -> text description
    "electric guitar": "distorted electric guitar",
    "piano":           "piano",
    "synthesizer":     "electronic synthesizer",
    "strings":         "orchestral string instruments",
    "acoustic guitar": "acoustic guitar",
    "drums":           "heavy drums and percussion",
    "voice":           "singing voice",
}

def clap_scores(y, groups):
    # Zero-shot CLAP: softmax within each group of labels, per 10 s window and averaged.
    from transformers import ClapModel, ClapProcessor
    model = ClapModel.from_pretrained(CLAP_MODEL).eval()
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    clap_sr = 48000
    y48 = librosa.resample(y, orig_sr=SR, target_sr=clap_sr)
    win = 10 * clap_sr
    windows = [y48[i:i + win] for i in range(0, max(1, len(y48) - win + 1), win)]
    texts = [t for g in groups.values() for t in g.values()]
    inputs = processor(text=texts, audio=windows, sampling_rate=clap_sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_audio                     # windows x texts
    out, per_window, i = {}, {}, 0
    for name, g in groups.items():
        probs = logits[:, i:i + len(g)].softmax(dim=-1).numpy()
        out[name] = dict(zip(g, probs.mean(axis=0)))
        per_window[name] = pd.DataFrame(probs, columns=list(g),
                                        index=[f"{10 * w}-{10 * w + 10} s" for w in range(len(windows))])
        i += len(g)
    return out, per_window

try:
    clap_out, clap_windows = clap_scores(y, {"vibe": VIBE_TEXT, "instruments": INSTRUMENTS})
    clap_vibe, instrument_scores = clap_out["vibe"], clap_out["instruments"]
    CLAP_OK = True
except Exception as e:
    print(f"CLAP unavailable ({type(e).__name__}: {e}); using the acoustic profile only")
    CLAP_OK, clap_windows = False, None
    clap_vibe = acoustic_scores
    stem_tag = {"drums": "drums", "other": "synthesizer", "vocals": "voice", "bass": "electric guitar"}
    instrument_scores = {k: 0.0 for k in INSTRUMENTS}
    for stem, tag in stem_tag.items():
        if stem_present[stem]:
            instrument_scores[tag] += stem_means[stem] / sum(stem_means.values())
top_instruments = sorted(instrument_scores, key=instrument_scores.get, reverse=True)[:2]

# ---- Combine the two audio opinions ------------------------------------------------------
w = VIBE_WEIGHTS
vibe_scores = {v: w["acoustic"] * acoustic_scores[v] + w["clap"] * clap_vibe[v] for v in VIBE_NAMES}
AUDIO_VIBE = max(vibe_scores, key=vibe_scores.get)

print(f"Key: {KEY} (r = {key_corr:.2f}) | dynamic range {dynamic_range_db:.1f} dB | drum share {DRUM_SHARE:.0%}")
pd.DataFrame({"acoustic profile": profile}).T.round(2)
""")

code(r"""
# ---- C. Lyrics: transcribe the vocal stem, then score its sentiment -----------------------
LYRICS, LYRIC_NEGATIVITY, LYRICS_NOTE, LYRICS_MOSTLY_NEGATIVE = "", None, "", False
if not LYRICS_ANALYSIS:
    LYRICS_NOTE = "skipped (LYRICS_ANALYSIS = False)"
elif not stem_present["vocals"]:
    LYRICS_NOTE = "skipped (no vocals detected)"
else:
    try:
        from transformers import pipeline
        from transformers.utils import logging as transformers_logging
        transformers_logging.set_verbosity_error()
        asr = pipeline("automatic-speech-recognition", model=WHISPER_MODEL, device=-1)   # clips are <= 30 s
        vocals_16k = librosa.resample(stems["vocals"], orig_sr=SR, target_sr=16000)
        LYRICS = asr({"raw": vocals_16k, "sampling_rate": 16000}, generate_kwargs={"task": "transcribe"})["text"].strip()
        sentiment = pipeline("sentiment-analysis", model=SENTIMENT_MODEL, device=-1, top_k=None)
        if LYRICS:
            probs = {d["label"].lower(): d["score"] for d in sentiment(LYRICS[:1000])[0]}
            LYRIC_NEGATIVITY = probs.get("negative", 0.0)          # probability the lyrics are negative
            LYRICS_MOSTLY_NEGATIVE = max(probs, key=probs.get) == "negative"
        LYRICS_NOTE = f"Whisper ({WHISPER_MODEL}) + sentiment ({SENTIMENT_MODEL})"
    except Exception as e:
        LYRICS_NOTE = f"failed ({type(e).__name__}: {e})"

BITTERSWEET = (AUDIO_VIBE in ("party", "happy") and LYRICS_MOSTLY_NEGATIVE
               and LYRIC_NEGATIVITY >= BITTERSWEET_NEGATIVITY)
VIBE = VIBE_OVERRIDE or ("bittersweet" if BITTERSWEET else AUDIO_VIBE)
VIBE_METHOD = (f"{w['acoustic']:.0%} acoustic profile + {w['clap']:.0%} CLAP" if CLAP_OK else "acoustic profile only") + \
              (" + lyrics rule" if BITTERSWEET else "")

print(f"Lyrics: {LYRICS_NOTE}")
if LYRICS:
    print(f'  transcript: "{LYRICS[:300]}{"..." if len(LYRICS) > 300 else ""}"')
    print(f"  lyric negativity (P(negative)): {LYRIC_NEGATIVITY:.2f}")
print(f"\nAudio vibe: {AUDIO_VIBE}  ->  FINAL VIBE: {VIBE}  ({VIBE_METHOD}{', overridden' if VIBE_OVERRIDE else ''})")

fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), gridspec_kw={"width_ratios": [1.1, 1.3, 1]})
ax = axes[0]
ax.barh(FEATURE_NAMES, [profile[f] for f in FEATURE_NAMES], color=COLORS["rms"], height=0.6)
ax.set(xlim=(0, 1), title="A. Acoustic profile (0-1)"); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax = axes[1]
x = np.arange(len(VIBE_NAMES))
for k, (name, sc, color) in enumerate([("acoustic", acoustic_scores, COLORS["mid"]), ("CLAP", clap_vibe, COLORS["onset"]),
                                       ("combined", vibe_scores, COLORS["rms"])]):
    ax.bar(x + (k - 1) * 0.27, [sc[v] for v in VIBE_NAMES], width=0.25, color=color, label=name)
ax.set_xticks(x, VIBE_NAMES, rotation=20)
ax.set(ylabel="probability", title=f"Vibe scores (final: {VIBE})"); ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
ax = axes[2]
names = list(instrument_scores)
ax.barh(names, [instrument_scores[n] for n in names],
        color=[COLORS["rms"] if n in top_instruments else "#c3c2b7" for n in names], height=0.6)
ax.set(xlim=(0, 1), title="Instruments / textures (CLAP)"); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(FIG_DIR / "vibe.png"); plt.show()

# How the vibe changes over time (CLAP per 10 s window) and the prototypes used by the acoustic rule
if clap_windows is not None:
    display(clap_windows["vibe"].round(2).style.background_gradient(axis=1, cmap="Blues")
            .set_caption("CLAP vibe per 10-second window"))
display(PROTOTYPES.style.format("{:.2f}").set_caption("Acoustic prototypes (expected feature values per vibe)"))
""")

# =============================================================================
# 9.2 Song signature
# =============================================================================
md(r"""
## 9.2 Song signature: shapes, materials and seed

So that different songs produce **different shapes**, not just different colours, three song-level properties feed every prompt:

| Property | Measured from | Becomes | Rule |
|---|---|---|---|
| **Number of star points** | tempo | `N-pointed star` | < 80 BPM → 5, < 100 → 6, < 120 → 8, < 140 → 10, else 12 |
| **Form** | key mode (from section 9.1) | major → *rounded petal shapes*, minor → *sharp angular shards* | |
| **Layout** | drum share of the stem energy ("rhythmic drive") | ≥ `DRIVE_THRESHOLD` → *concentric tunnel rings*, else *swirling spiral arms* | |
| **Materials** | top 2 CLAP instrument / texture tags (section 9.1) | e.g. distorted guitar → *jagged shattered glass*, piano → *smooth ivory curves* | table below |
| **Image seed** | fingerprint (SHA-256) of the clip's audio samples | `SONG_SEED + section` | same song → same images; different song → different compositions |

Every option stays a **centred radial composition** (a star inside a star), so the part-by-part animation in section 13 still works.
These mappings are design choices (cross-modal metaphors), not perceptual facts.
""")

code(r"""
# 1. Seed from the song itself: a fingerprint of the audio samples
SONG_SEED = int(hashlib.sha256(np.round(y, 4).tobytes()).hexdigest()[:8], 16) % 1_000_000

# 2. Shape family (KEY / MODE and DRUM_SHARE were measured in section 9.1)
N_POINTS = next(n for limit, n in [(80, 5), (100, 6), (120, 8), (140, 10), (np.inf, 12)] if tempo < limit)
FORM = "rounded petal shapes" if MODE == "major" else "sharp angular shards"
LAYOUT = "concentric tunnel rings" if DRUM_SHARE >= DRIVE_THRESHOLD else "swirling spiral arms"

# 3. Materials from the two strongest instrument / texture tags
MATERIALS = {
    "electric guitar": "jagged shattered glass",
    "piano":           "smooth ivory curves",
    "synthesizer":     "glowing circuit traces",
    "strings":         "flowing silk ribbons",
    "acoustic guitar": "carved wood grain",
    "drums":           "hammered metal plates",
    "voice":           "soft smoke wisps",
}
MATERIAL_TEXT = " and ".join(MATERIALS[t] for t in top_instruments)

SHAPE_TEXT = f"{N_POINTS}-pointed star inside a larger star, {LAYOUT}, {FORM}"
signature = pd.DataFrame([
    ("tempo", f"{tempo:.1f} BPM", f"{N_POINTS}-pointed star"),
    ("key / mode", f"{KEY} (r = {key_corr:.2f})", FORM),
    ("drum share of stems", f"{DRUM_SHARE:.0%}", LAYOUT),
    ("top instruments (CLAP)", ", ".join(top_instruments), MATERIAL_TEXT),
    ("vibe", VIBE, "theme / palette (section 10)"),
    ("audio fingerprint", f"seed {SONG_SEED}", f"section seeds {SONG_SEED + 1}..{SONG_SEED + n_sections}"),
], columns=["property", "measured", "visual result"])
signature.to_csv(OUTPUT_DIR / "song_signature.csv", index=False)
signature
""")

# =============================================================================
# 9.3 Dramatic beat changes
# =============================================================================
md(r"""
## 9.3 Detect dramatic beat changes → completely new image

Normally each section's image is generated *from the previous section's image* (img2img, section 11), so the artwork evolves smoothly.
When the **beat changes dramatically** (a drop, the drums cutting out, a switch to half-time) the video should change completely instead.

For every section boundary, the notebook compares the `DRAMATIC_WINDOW_SEC` = 3 s **before** with the 3 s **after**:

| Change | Measured as | Weight |
|---|---|---|
| drum level | difference in mean drum-stem level (0–1) | 1.0 |
| drum-hit rate | relative difference in drum hits per second | 1.0 |
| bass level | difference in mean bass-stem level (0–1) | 1.0 |
| local tempo | \|log2(tempo after / tempo before)\|, capped at 1 (= half or double time) | 0.5 (tempo estimates on 3 s are noisy) |

The **change score is the largest weighted change**, not the sum: a drop where only the kick and bass disappear is dramatic even if the
hi-hats keep the same hit rate, and averaging would dilute it. If the **change score ≥ `DRAMATIC_THRESHOLD`** (0.50), that section:

1. **restarts the image chain:** generated from text with a fresh seed, not from the previous image;
2. gets a **completely different shape family**, chosen from the new section's own rhythm:

| New section's rhythm | Shape family |
|---|---|
| heavy drums + many hits | concentric hexagon **tunnel** |
| heavy drums, fewer hits | radiating **sunburst** of light rays |
| light drums, bright highs | faceted **crystal** geode |
| light drums, strong mids | spiral **vortex** galaxy |
| light drums, quiet | flowing **waves** of light (full-frame, not centred) |
| otherwise | blooming **flower** |

   (if that equals the current shape, the next family in the list is used, so the shape always changes);
3. **cuts** to the new image in `DRAMATIC_CROSSFADE` = 0.25 s with a flash, instead of the 1 s crossfade.

Each family also has its **own motion style** (section 12), so a dramatic change switches both what you see and how it moves. Weights and threshold are design choices, checked
on the synthesized demo track, where the true changes are known (kick + bass enter at 8 s and drop out at 24 s; hi-hats join at 16 s).
""")

code(r"""
local_tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=SR, hop_length=HOP_LENGTH, aggregate=None)

def window_mean(values, times, a, b):
    m = (times >= a) & (times < b)
    return float(values[m].mean()) if m.any() else 0.0

def hit_rate(a, b):
    return np.sum((drum_times >= a) & (drum_times < b)) / max(b - a, 1e-9)

FAMILY_ORDER = ["tunnel", "sunburst", "crystal", "vortex", "flower", "waves"]
SHAPE_FAMILIES = {
    "star":     f"{N_POINTS}-pointed star inside a larger star, {LAYOUT}",      # the song-level shape (section 9.2)
    "tunnel":   "concentric hexagon tunnel receding to a bright center",
    "sunburst": "radiating sunburst of long light rays",
    "crystal":  "faceted crystal geode with a glowing core",
    "vortex":   "spiral vortex galaxy swirling into a bright center",
    "flower":   "blooming flower with layered petals",
    "waves":    "flowing layered waves of light across the frame",
}

def pick_family(sec_id, avoid):
    # Shape family from the new section's own rhythm; never the same as the current one.
    part = timeline[timeline.section_id == sec_id]
    row = sections[sections.section == sec_id].iloc[0]
    drums = part.drums.mean()
    if drums >= 0.5 and row.onset_density_rel >= 0.6:
        family = "tunnel"
    elif drums >= 0.5:
        family = "sunburst"
    elif row.high >= 0.6:
        family = "crystal"
    elif row.mid >= 0.5:
        family = "vortex"
    elif row.rms < 0.4:
        family = "waves"
    else:
        family = "flower"
    if family == avoid:
        family = FAMILY_ORDER[(FAMILY_ORDER.index(family) + 1) % len(FAMILY_ORDER)]
    return family

W, wd = DRAMATIC_WINDOW_SEC, DRAMATIC_WEIGHTS
change_rows, current = [], "star"
plan = {1: {"family": "star", "new_image": True}}
for k, b in enumerate(section_edges[1:-1], start=2):          # boundary before section k
    before, after = (max(0.0, b - W), b), (b, min(DURATION, b + W))
    d_drums = abs(window_mean(timeline.drums.values, frame_times, *after) - window_mean(timeline.drums.values, frame_times, *before))
    r0, r1 = hit_rate(*before), hit_rate(*after)
    d_hits = abs(r1 - r0) / max(r0, r1, 1e-9)
    d_bass = abs(window_mean(timeline.bass_stem.values, frame_times, *after)
                 - window_mean(timeline.bass_stem.values, frame_times, *before))
    t0 = np.median(local_tempo[(feat_times >= before[0]) & (feat_times < before[1])])
    t1 = np.median(local_tempo[(feat_times >= after[0]) & (feat_times < after[1])])
    d_tempo = min(1.0, abs(np.log2(t1 / t0)))
    score = max(wd["drums"] * d_drums, wd["drum_hits"] * d_hits, wd["bass"] * d_bass, wd["tempo"] * d_tempo)
    dramatic = score >= DRAMATIC_THRESHOLD
    if dramatic:
        current = pick_family(k, avoid=current)
    plan[k] = {"family": current, "new_image": dramatic}
    change_rows.append({"boundary_sec": round(b, 2), "into_section": k, "d_drums": d_drums, "d_drum_hits": d_hits,
                        "d_bass": d_bass, "tempo_before": t0, "tempo_after": t1, "change_score": score,
                        "dramatic": dramatic, "shape_after": current})

beat_changes = pd.DataFrame(change_rows)
DRAMATIC_BOUNDARIES = beat_changes.loc[beat_changes.dramatic, "boundary_sec"].to_numpy(float)
section_crossfades = np.where(beat_changes.dramatic, DRAMATIC_CROSSFADE, CROSSFADE_SECONDS) if len(beat_changes) else []
beat_changes.to_csv(OUTPUT_DIR / "beat_changes.csv", index=False)
print(f"Dramatic beat changes at {DRAMATIC_BOUNDARIES.tolist()} s (threshold {DRAMATIC_THRESHOLD})")
beat_changes.round(2)
""")

code(r"""
fig, ax = plt.subplots(figsize=(12, 3.2))
ax.plot(frame_times, timeline.drums, color=COLORS["drums"], lw=1.0, label="drum stem level")
ax.plot(frame_times, timeline.bass_stem, color=COLORS["bass_stem"], lw=1.0, label="bass stem level")
ax.vlines(drum_times, 0, 0.08, color=COLORS["beat"], lw=0.6, label="drum hits")
for _, r in beat_changes.iterrows():
    style = dict(color=COLORS["baseline"], lw=2.2, ls="-") if r.dramatic else dict(color="#52514e", lw=1.0, ls="--")
    ax.axvline(r.boundary_sec, **style)
    ax.text(r.boundary_sec + 0.15, 1.02, f"{r.change_score:.2f}" + (f" → {r.shape_after}" if r.dramatic else ""),
            fontsize=9, color=style["color"])
ax.set(title="Section boundaries: solid red = dramatic beat change (new image + new shape), dashed = smooth evolution",
       xlabel="time (s)", ylabel="level (0-1)", xlim=(0, DURATION), ylim=(0, 1.12))
ax.legend(loc="upper left", fontsize=8, ncol=3)
plt.tight_layout(); plt.savefig(FIG_DIR / "beat_changes.png"); plt.show()
""")

# =============================================================================
# 10. Prompts
# =============================================================================
md(r"""
## 10. Build GenAI prompts from section features

The diffusion model never sees the waveform. Instead each section goes through a transparent chain:

`vibe + section statistics → low / medium / high levels → visual descriptors → text prompt → image`

Every prompt has three parts:

1. **Structure:** the section's **shape family** (section 9.3: the song's *N-pointed star* until a dramatic beat change switches it to a
   tunnel, sunburst, crystal, vortex or flower) plus the song's form and materials (section 9.2). A centred, layered radial structure is required because the renderer (section 13) animates the
   image **part by part by radius** (core, inner star, outer star).
2. **Vibe theme** (same for all sections, from section 9.1): colours and lighting, e.g. *neon club lights* for party.
3. **Section descriptors** (different per section): one short phrase each for energy, bass, mid and high-frequency content.
   Spectral brightness and onset density are still measured and shown in `section_description`, but left out of the prompt to stay
   under the token limit (the song-level shape words carry more information).

**Levels** (documented design thresholds): `0.00–0.33 → low`, `0.33–0.67 → medium`, `0.67–1.00 → high`.

**Wavelength-inspired visual grammar.** Low-frequency sound has longer physical wavelengths and high-frequency sound shorter ones, so the
mapping uses *large, broad* forms for bass and *small, fine* details for highs. This is a cross-modal **design choice**; the program
measures frequency, and the images do not physically depict acoustic wavelength.

**Prompt length.** SD-Turbo's CLIP text encoder reads at most **77 tokens**; anything beyond is silently cut off, so phrases are kept
short and a token check confirms nothing is truncated. The full readable sentence is stored separately in `section_description`.
(With `guidance_scale = 0`, SD-Turbo ignores negative prompts, so "no text / no people" is enforced by keeping the vocabulary abstract.)
""")

code(r"""
def level(x):
    return "low" if x < 1 / 3 else ("medium" if x < 2 / 3 else "high")

# 1. Song structure shared by every section (the renderer animates these parts)
def structure(section):
    # Shape family for this section (section 9.3) + the song's form and materials (section 9.2)
    shape = SHAPE_FAMILIES[plan[section]["family"]]
    layout = "full-frame composition" if plan[section]["family"] == "waves" else "centered radial composition"
    art = "psychedelic kaleidoscope art" if VISUAL_STYLE == "psychedelic" else "luminous digital art"
    return f"{art}, {layout}, {shape}, {FORM}, made of {MATERIAL_TEXT}, dark background"

# 2. Vibe theme shared by every section
VIBE_THEMES = {
    "party":       "neon club lights, laser beams, magenta and cyan glow",
    "happy":       "sunny warm light, golden and coral glow",
    "chill":       "soft mist, pastel teal light, gentle glow",
    "melancholic": "deep blue haze, muted indigo light, rain",
    "dark":        "deep red and black, smoke, harsh light",
    "epic":        "golden light rays, vast cosmic scale",
    "bittersweet": "warm amber glow fading into cold blue shadow",
}

# 3. One short descriptor per feature and level
DESCRIPTORS = {
    "rms":      {"low": "sparse dim calm", "medium": "layered moderately bright", "high": "dense intense glow"},
    "bass":     {"low": "small central core", "medium": "round central core", "high": "huge glowing core"},
    "mid":      {"low": "simple layers", "medium": "layered points", "high": "many layered points"},
    "high":     {"low": "smooth surfaces", "medium": "some fine lines", "high": "fine sparkling particles"},
    "centroid": {"low": "soft rounded edges", "medium": "balanced edges", "high": "crisp sharp edges"},
    "onset_density_rel": {"low": "few elements", "medium": "several accents", "high": "fragmented bursts"},
}
LEVEL_FEATURES = ["rms", "bass", "mid", "high", "centroid", "onset_density_rel"]   # all measured + described
PROMPT_FEATURES = ["rms", "bass", "mid", "high"]                                  # the ones sent to the model

def build_prompt(row):
    levels = {f: level(row[f]) for f in LEVEL_FEATURES}
    phrases = [DESCRIPTORS[f][levels[f]] for f in PROMPT_FEATURES]
    prompt = ", ".join([structure(int(row.section)), VIBE_THEMES[VIBE]] + phrases)
    description = (f"A {VIBE} song. This section has {levels['rms']} energy with {levels['bass']} low-frequency presence, "
                   f"{levels['mid']} mid-frequency activity, {levels['high']} high-frequency detail, "
                   f"{levels['centroid']} spectral brightness and {levels['onset_density_rel']} onset density.")
    return prompt, description, levels

records = []
for _, row in sections.iterrows():
    prompt, description, levels = build_prompt(row)
    s = int(row.section)
    first = plan[s]["new_image"]                  # section 1, or a dramatic beat change -> start a new image
    records.append({"section": s, "vibe": VIBE, "shape_family": plan[s]["family"],
                    **{f"{k}_level": v for k, v in levels.items()},
                    "section_description": description, "prompt": prompt,
                    "seed": SONG_SEED + s, "model": MODEL_ID, "guidance_scale": 0.0,
                    "mode": "text2img" if first else "img2img",
                    "init_image": "" if first else str(IMAGE_DIR / f"section_{s - 1}.png"),
                    "strength": 1.0 if first else IMG2IMG_STRENGTH,
                    "steps": GEN_STEPS if first else IMG2IMG_STEPS,
                    "size": f"{IMAGE_SIZE}x{IMAGE_SIZE}", "image_path": str(IMAGE_DIR / f"section_{s}.png")})
prompts = pd.DataFrame(records)

for _, r in prompts.iterrows():
    print(f"Section {r.section}: {r.section_description}\n  PROMPT: {r.prompt}\n")
""")

code(r"""
# Token-count check: SD-Turbo's CLIP text encoder truncates after 77 tokens (incl. start/end tokens)
try:
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    prompts["n_tokens"] = [len(tokenizer(p).input_ids) for p in prompts.prompt]
    print("Tokens per prompt:", prompts.n_tokens.tolist(), "(limit 77)")
    assert prompts.n_tokens.max() <= 77, "a prompt would be truncated; shorten descriptors"
except OSError as e:
    print("Tokenizer not available offline, skipping token check:", e)

prompts[["section", "vibe", "shape_family", "rms_level", "bass_level", "mid_level", "high_level", "centroid_level", "onset_density_rel_level"]]
""")

# =============================================================================
# 11. Generate images
# =============================================================================
md(r"""
## 11. Generate (or load cached) section images

Model: **[SD-Turbo](https://huggingface.co/stabilityai/sd-turbo)** (distilled from Stable Diffusion 2.1, built for 1–4 steps at 512×512).
It is light enough for a free Colab T4 or an Apple-silicon laptop. Before publishing results, check the model card's license terms
(SD-Turbo is released under the Stability AI community license).

**Image-to-image chaining:** Section 1, and any section that follows a **dramatic beat change** (section 9.3), is generated from text
alone, with a new shape family. Every other section starts from the **previous section's image** (img2img,
`strength = IMG2IMG_STRENGTH` = 0.6) with its own prompt, so between dramatic changes the artwork evolves smoothly: the composition
carries over while energy, bass and detail change with the music.

**Caching:** each image is saved to `outputs/images/section_N.png` together with `outputs/prompts.csv`. On later runs an image is only
regenerated if its prompt or settings changed (or `FORCE_REGENERATE = True`). Because of the chaining, regenerating one section also
regenerates every section after it.

**Seeds:** section N uses `SONG_SEED + N`, where `SONG_SEED` is a fingerprint of the clip (section 9.2): re-running a song reproduces
its images, while a different song starts from different noise.

If diffusion is unavailable (no `diffusers` / download fails), a clearly labelled **placeholder gradient** is used so the rest of the
pipeline still runs. Placeholders are *not* valid project results.
""")

code(r"""
PROMPTS_CSV = OUTPUT_DIR / "prompts.csv"
GEN_KEYS = ["prompt", "seed", "model", "steps", "size", "mode", "init_image", "strength"]

def needs_generation(r, previous):
    # True if the image is missing or was generated with different prompt/settings.
    if FORCE_REGENERATE or not Path(r.image_path).exists() or previous is None:
        return True
    old = previous[previous.section == r.section]
    return old.empty or any(str(old.iloc[0][k]) != str(r[k]) for k in GEN_KEYS)

def placeholder_image(section, size=IMAGE_SIZE):
    # Radial gradient fallback, used only if diffusion is not available.
    yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    r = np.sqrt(xx ** 2 + yy ** 2)
    hue = [0.75, 0.55, 0.15, 0.95][(section - 1) % 4]
    base = np.stack([0.5 + 0.5 * np.cos(2 * np.pi * (hue + o)) for o in (0, 0.33, 0.67)])
    img = np.clip((1 - r)[..., None] * base, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))

previous = pd.read_csv(PROMPTS_CSV, keep_default_na=False) if PROMPTS_CSV.exists() else None
stale = [int(r.section) for _, r in prompts.iterrows() if needs_generation(r, previous)]
# Chaining: once a section changes, every later section (which starts from it) must be regenerated too
todo = [r for _, r in prompts.iterrows() if stale and r.section >= min(stale)]
print(f"{len(prompts) - len(todo)} cached image(s), {len(todo)} to generate")

prompts["generated_by"] = "cache"
if previous is not None and "generated_by" in previous:
    prompts["generated_by"] = prompts.section.map(previous.set_index("section").generated_by).fillna("cache")

if todo:
    try:
        from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
        from diffusers.utils import logging as diffusers_logging
        from transformers.utils import logging as transformers_logging
        diffusers_logging.set_verbosity_error(); transformers_logging.set_verbosity_error()
        dtype = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None).to(DEVICE)
        pipe.set_progress_bar_config(disable=True)
        img2img = AutoPipelineForImage2Image.from_pipe(pipe)          # shares the same weights
        img2img.set_progress_bar_config(disable=True)
        for r in todo:
            generator = torch.Generator("cpu").manual_seed(int(r.seed))
            if r["mode"] == "text2img":   # (r.mode would be the pandas method)
                image = pipe(r.prompt, num_inference_steps=int(r.steps), guidance_scale=0.0,
                             width=IMAGE_SIZE, height=IMAGE_SIZE, generator=generator).images[0]
            else:
                init = Image.open(r.init_image).convert("RGB")
                image = img2img(r.prompt, image=init, num_inference_steps=int(r.steps), strength=float(r.strength),
                                guidance_scale=0.0, generator=generator).images[0]
            image.save(r.image_path)
            prompts.loc[prompts.section == r.section, "generated_by"] = MODEL_ID
            print(f"  generated section {r.section} ({r['mode']})")
        del pipe, img2img
    except Exception as e:
        print(f"Diffusion unavailable ({type(e).__name__}: {e}).\nUsing PLACEHOLDER images - not valid results.")
        for r in todo:
            placeholder_image(r.section).save(r.image_path)
            prompts.loc[prompts.section == r.section, "generated_by"] = "PLACEHOLDER"

prompts.to_csv(PROMPTS_CSV, index=False)
section_images = [np.asarray(Image.open(p).convert("RGB").resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS),
                             dtype=np.float32) / 255.0 for p in prompts.image_path]

fig, axes = plt.subplots(1, n_sections, figsize=(4 * n_sections, 4.4))
for ax, img, (_, r) in zip(axes, section_images, prompts.iterrows()):
    ax.imshow(img); ax.axis("off")
    ax.set_title(f"Section {r.section}: {r.shape_family} ({r['mode']})\nenergy {r.rms_level} · bass {r.bass_level} · "
                 f"high {r.high_level}", fontsize=10)
plt.tight_layout(); plt.savefig(FIG_DIR / "section_images.png"); plt.show()
""")

# =============================================================================
# 11.1 Layers
# =============================================================================
md(r"""
## 11.1 Split each generated image into movable parts

To move the **parts** of an image (not just the whole picture), every section image is decomposed once into:

| Part | How it is found | Why it works |
|---|---|---|
| **Core**, **inner star**, **outer star** | soft radial masks: radius < 0.24, 0.24–0.58, > 0.58 of the half-frame | the prompts ask for a centred star inside a larger star, so the parts sit at different radii |
| **Line art** | Canny edge detection on a blurred copy of the image (main structural lines only) | the image's own lines become a separate glowing layer |
| **Light spots** | pixels brighter than the image's 92nd luminance percentile | the image's own bright spots act as "stage lights" |

The masks are soft (they fade over `mask_softness`), so neighbouring parts blend when they rotate at different speeds; this is
what produces the visible *twist* between the inner and outer star.
""")

code(r"""
S = OUTPUT_SIZE
yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
CX = CY = (S - 1) / 2
RADIUS = np.hypot(xx - CX, yy - CY) / (S / 2)          # 0 at the centre, 1 at the frame edge
THETA = np.arctan2(yy - CY, xx - CX)

def smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)

def part_weights(r, core_radius, inner_radius, soft=VIS["mask_softness"]):
    # Soft weights (sum to 1) for core / inner star / outer star at radius r.
    to_inner = smoothstep(core_radius - soft, core_radius + soft, r)
    to_outer = smoothstep(inner_radius - soft, inner_radius + soft, r)
    return 1 - to_inner, to_inner * (1 - to_outer), to_outer

def decompose(img):
    # Line-art layer and light-spot layer of one section image (float RGB in [0, 1]).
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # Blur first so only the main structural lines survive, not every fine texture detail
    edges = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 2.0), 60, 140)
    lines = cv2.GaussianBlur(edges.astype(np.float32) / 255, (0, 0), 1.0)
    lines /= lines.max() + 1e-6
    lum = gray.astype(np.float32) / 255
    lights = cv2.GaussianBlur((lum > np.percentile(lum, 92)).astype(np.float32), (0, 0), 3.0)
    return {"lines": lines, "lights": lights}

layers = [decompose(img) for img in section_images]

# Show the decomposition for every section
w_core, w_inner, w_outer = part_weights(RADIUS, VIS["core_radius"], VIS["inner_radius"])
part_view = np.stack([w_core, w_inner, w_outer], axis=-1)            # red = core, green = inner, blue = outer
fig, axes = plt.subplots(n_sections, 4, figsize=(11, 2.8 * n_sections))
for k in range(n_sections):
    for ax, im, title in zip(axes[k], [section_images[k], 0.5 * section_images[k] + 0.5 * part_view,
                                       layers[k]["lines"], layers[k]["lights"]],
                             ["image", "parts (core / inner / outer)", "line art", "light spots"]):
        ax.imshow(im, cmap="gray" if im.ndim == 2 else None); ax.axis("off")
        ax.set_title(f"S{k + 1}: {title}", fontsize=9)
plt.tight_layout(); plt.savefig(FIG_DIR / "layers.png"); plt.show()
""")

# =============================================================================
# 12. Mappings
# =============================================================================
md(r"""
## 12. Define audio-to-visual mappings

Each part of the image follows a different musical signal (coefficients in `VIS`). Large, slow parts follow low frequencies; small, fast
details follow high frequencies (the wavelength-inspired grammar from section 10):

| Image part | Driven by | Motion | Formula (per frame) |
|---|---|---|---|
| **Core** | bass stem, beat | grows / shrinks | `scale = 1 + 0.22·bass_stem + 0.10·beat` |
| **Inner star** | lead (vocals, or mid band if no vocals) | rotates clockwise and **twists** | `speed = 8 + 80·lead` °/s, `twist = 1.6·lead` rad |
| **Outer star** | other stem (synths, guitars, keys) | rotates counter-clockwise, pulses on beats | `speed = 5 + 45·other` °/s, `scale = 1 + 0.05·beat` |
| **Whole image** | high band | fine fast ripple | `ripple = 0.012·high` × sin(7 waves) |
| **Line art** | drum hits | flashes and stretches outward | `glow = 0.05 + 0.55·drum_pulse` |
| **Light spots** | drum hits, RMS | glow, change colour on every hit | `gain = 0.15 + 0.65·drum_pulse + 0.10·rms` |
| Particles | high band | more particles | `n = 30 + 250·high` |
| Brightness / saturation | RMS | overall intensity | `brightness = 0.70 + 0.25·rms` |
| Sharpness | spectral centroid | crisper when brighter | `sharpness = 0.90 + 0.30·centroid` |
| White flash | onsets | brief flash | `flash = 0.12·onset_pulse` |
| Section image | section id, dramatic beat change | crossfade / cut | 1 s crossfade; **0.25 s cut + flash** at a dramatic beat change |

### Psychedelic style (default, `VISUAL_STYLE = "psychedelic"`)

| Effect | What happens | Driven by |
|---|---|---|
| **Endless zoom** | the view keeps zooming into the image forever; two zoom levels one octave apart are blended so the loop never jumps, and the radius is mirrored so the pattern repeats in concentric rings | speed = 0.05 + 0.15·RMS + **0.9·beat pulse** octaves/s (surges forward on every beat) |
| **Always kaleidoscope** | the image is folded into N mirrored wedges (N = the song's star points) | wedge rotation: mid band; pattern slides through the wedges: other stem |
| **Morph transitions** | the next image *melts in*, its brightest parts first, while a swirl twists both images (peak at the middle of the transition); no hard cuts and no flash | 2.5 s normally, 1.5 s at a dramatic beat change |
| **Colour cycling** | the whole frame rotates through the rainbow; saturation pops on every beat | hue speed: RMS; hue jump on every drum hit |
| **Beat zoom punch** | on every beat the view punches in by 10 % and eases back out (on top of the endless zoom) | beat pulse |
| **Instruments in the design** | core / inner / outer lines glow orange / magenta / cyan as loud as bass / vocals / other; a ring blooms when its instrument enters; the number of mirror segments changes when instruments enter or leave | stems + instrument events (section 7.1) |

**No cuts:** in this style every beat, drum-hit and onset pulse is softened so it swells in over ~2 frames (peak still on the beat),
colour steps on drum hits glide in as the hit decays, and onset flashes are at 40 % strength. One-frame jumps read as cuts; these
keep the motion continuous while staying in time with the music.

Line art, light spots, particles, brightness and sharpness work as in the table above.

### Motion modes (`VISUAL_STYLE = "modes"`): every shape family moves differently

Spinning rings are only one of seven motion styles. Each section's image moves according to its shape family (section 9.3), and the
same audio controls drive each style in its own way:

| Shape family | Motion mode | What moves | Driven by |
|---|---|---|---|
| star (song shape) | **kaleidoscope** | image folded into N mirrored wedges (N = the song's star points) that slide through the pattern | other stem (slide), mid (fold rotation), bass (scale) |
| tunnel | **zoom** | flies *into* the image over the section, punching forward on beats | bass + beats |
| sunburst | **shockwave** | a ring wave travels outward from the centre on every beat | beats, bass |
| crystal | **shatter** | the image breaks into tiles that jump apart on every drum hit, then snap back | drum hits |
| vortex | **spin** | inner star twists clockwise, outer star counter-rotates, core pulses (the original mode) | lead, other stem, bass |
| flower | **sway** | petals rock back and forth instead of spinning; the core blooms | lead, bass |
| waves | **flow** | the whole image ripples like liquid, sideways and up | lead (speed + depth), RMS |

All modes add a slow **camera drift** (speed from the lead voice) so nothing stays locked to the centre. Particles burst outward in the
radial modes and rise like sparks in *sway* and *flow*. During a crossfade, each of the two images keeps its own motion mode.
The line-art and light-spot layers use the same warp as their image, so they stay attached to the artwork.

Rotation angles are integrated over time (`cumsum(speed) / FPS`) so speed changes never cause jumps. Both the audio-conditioned version
and the baseline produce a **parameter table with identical columns**, and one renderer draws both.
""")

code(r"""
def crossfade_plan(times, edges, crossfades=CROSSFADE_SECONDS):
    # For each frame: index of image A, index of image B and blend weight (0 = all A, 1 = all B).
    # `crossfades` is one duration for every boundary, or one per boundary.
    sec = np.clip(np.searchsorted(edges, times, side="right") - 1, 0, len(edges) - 2)
    img_a, img_b, blend = sec.copy(), sec.copy(), np.zeros_like(times)
    for k, (b, crossfade) in enumerate(zip(edges[1:-1], np.broadcast_to(crossfades, len(edges) - 2))):
        in_fade = np.abs(times - b) < crossfade / 2
        img_a[in_fade], img_b[in_fade] = k, k + 1
        blend[in_fade] = (times[in_fade] - (b - crossfade / 2)) / crossfade
    return img_a, img_b, blend


def visual_params(ctrl, edges, crossfades=CROSSFADE_SECONDS, flash_times=()):
    # ctrl: DataFrame of controls (all 0-1 except drum_hits). Returns one row of visual parameters per frame.
    v, t = VIS, ctrl.time_sec.values
    deg_per_frame = lambda speed: np.cumsum(speed) / FPS           # integrate deg/s -> degrees
    p = pd.DataFrame({"time_sec": t})
    # Event pulses. In the psychedelic style they are softened with a short centred window, so every hit swells in
    # over ~0.1 s instead of jumping in one frame (a one-frame jump reads as a cut); the peak stays on the beat.
    pulse = {c: ctrl[c].values.astype(float) for c in ["beat_pulse", "drum_pulse", "onset_pulse"]}
    if VISUAL_STYLE == "psychedelic":
        win = np.hanning(max(3, int(round(v["soft_attack_sec"] * FPS)) + 2))[1:-1]
        pulse = {c: np.convolve(x, win / win.sum(), mode="same") for c, x in pulse.items()}
    beat, drum, onset = pulse["beat_pulse"], pulse["drum_pulse"], pulse["onset_pulse"]
    # Colour steps on drum hits are eased: +1 step glides in as the hit decays, instead of jumping
    drum_steps = ctrl.drum_hits.values - ctrl.drum_unit.values
    p["img_a"], p["img_b"], p["blend"] = crossfade_plan(t, edges, crossfades)
    # Progress (0-1) through each displayed image's own section, used by the zoom mode
    starts, lengths = np.asarray(edges[:-1]), np.diff(edges)
    p["age_a"] = np.clip((t - starts[p.img_a]) / lengths[p.img_a], 0, 1)
    p["age_b"] = np.clip((t - starts[p.img_b]) / lengths[p.img_b], 0, 1)
    # Zoom: slight bass push, plus a zoom PUNCH on every beat (+10%, easing back out)
    p["zoom"] = (v["zoom_base"] + v["zoom_bass"] * ctrl.bass.values) * (1 + v["beat_zoom_amp"] * beat)
    # Instruments inside the design: per-ring levels, entrance blooms, and kaleidoscope segment count
    p["lvl_bass"], p["lvl_lead"], p["lvl_other"] = ctrl.bass_stem.values, ctrl.lead.values, ctrl.other.values
    for n in ["bass", "vocals", "other", "drums"]:
        p[f"enter_{n}"] = ctrl[f"enter_{n}"].values
    target = np.clip(N_POINTS + 2 * (np.asarray(ctrl.active_count, int) - 2), 4, 16) * np.ones(len(t), int)
    fold_a, fold_b, fold_mix = target.copy(), target.copy(), np.zeros(len(t))
    n_morph, current, previous, changed = max(1, int(FOLD_MORPH_SEC * FPS)), target[0], target[0], -10 ** 9
    for i in range(len(t)):
        if target[i] != current:
            previous, current, changed = current, target[i], i
        if i - changed < n_morph:               # morph from the old to the new number of segments
            fold_a[i], fold_b[i], fold_mix[i] = previous, current, smoothstep(0, 1, (i - changed) / n_morph)
        else:
            fold_a[i] = fold_b[i] = current
    p["fold_a"], p["fold_b"], p["fold_mix"] = fold_a, fold_b, fold_mix
    p["core_scale"] = 1 + v["core_scale_bass"] * ctrl.bass_stem.values + v["core_scale_beat"] * beat
    p["core_angle"] = deg_per_frame(np.full(len(t), v["core_rot_base"]))
    p["inner_angle"] = deg_per_frame(v["inner_rot_base"] + v["inner_rot_lead"] * ctrl.lead.values)
    p["inner_twist"] = v["swirl_lead"] * ctrl.lead.values
    p["outer_angle"] = -deg_per_frame(v["outer_rot_base"] + v["outer_rot_other"] * ctrl.other.values)
    p["outer_scale"] = 1 + v["outer_scale_beat"] * beat
    p["ripple_amp"] = v["ripple_high"] * ctrl.high.values
    p["ripple_phase"] = 2 * np.pi * v["ripple_speed"] * t
    p["lines_gain"] = v["lines_base"] + v["lines_drums"] * drum
    p["lines_stretch"] = 1 + v["lines_stretch"] * drum
    p["lights_gain"] = v["lights_base"] + v["lights_drums"] * drum + v["lights_rms"] * ctrl.rms.values
    p["light_hue"] = (v["light_hue_step"] * drum_steps) % 1.0
    p["n_particles"] = (v["particles_min"] + v["particles_range"] * ctrl.high.values).astype(int)
    p["jitter"] = ctrl.high.values
    p["brightness"] = v["brightness_base"] + v["brightness_rms"] * ctrl.rms.values
    p["saturation"] = v["saturation_base"] + v["saturation_rms"] * ctrl.rms.values
    p["sharpness"] = v["sharpness_base"] + v["sharpness_centroid"] * ctrl.centroid.values
    # Motion-mode controls
    lead = ctrl.lead.values
    drift_phase = 2 * np.pi * np.cumsum(v["drift_speed_base"] + v["drift_speed_lead"] * lead) / FPS
    p["drift_x"] = v["drift_amp"] * np.sin(drift_phase)
    p["drift_y"] = v["drift_amp"] * np.sin(1.3 * drift_phase + 1.0)
    sway_phase = 2 * np.pi * np.cumsum(v["sway_speed_base"] + v["sway_speed_lead"] * lead) / FPS
    p["sway_angle"] = v["sway_deg"] * (0.4 + 0.6 * lead) * np.sin(sway_phase)
    p["beat_pulse"] = beat
    p["beat_age"] = np.minimum(-np.log(np.maximum(ctrl.beat_pulse.values, 1e-6)) / PULSE_DECAY, 2.0)  # s since beat
    p["drum_pulse"] = drum
    p["tile_seed"] = ctrl.drum_hits.values.astype(int)
    p["flow_amp"] = v["flow_amp_base"] + v["flow_amp_lead"] * lead + 0.01 * ctrl.rms.values
    p["flow_phase"] = 2 * np.pi * np.cumsum(v["flow_speed_base"] + v["flow_speed_lead"] * lead) / FPS
    # Psychedelic style: endless-zoom phase (octaves), kaleidoscope rotation, colour-cycling hue
    p["zoom_phase"] = np.cumsum(v["zoom_speed_base"] + v["zoom_speed_rms"] * ctrl.rms.values
                                + v["zoom_speed_beat"] * beat) / FPS
    p["kaleido_angle"] = deg_per_frame(v["kaleido_rot_base"] + v["kaleido_rot_mid"] * ctrl.mid.values)
    p["hue_shift"] = (np.cumsum(v["hue_speed_base"] + v["hue_speed_rms"] * ctrl.rms.values) / FPS
                      + v["hue_jump_drum"] * drum_steps) % 1.0
    p["flash"] = v["flash_onset"] * onset * (v["psy_flash_scale"] if VISUAL_STYLE == "psychedelic" else 1.0)
    for ft in flash_times:                                           # extra flash at dramatic beat changes
        p["flash"] += DRAMATIC_FLASH * np.where(t >= ft, np.exp(-PULSE_DECAY * (t - ft)), 0.0)
    return p

audio_params = visual_params(timeline, section_edges, crossfades=section_crossfades, flash_times=DRAMATIC_BOUNDARIES)
audio_params.describe().T[["mean", "min", "max"]].round(3)
""")

# =============================================================================
# 13. Renderer
# =============================================================================
md(r"""
## 13. Render the audio-conditioned visualizer

For every frame the renderer reads **one row of the parameter table** (looked up by frame timestamp). In the default **psychedelic**
style, step 1 is the endless kaleidoscope zoom with morph transitions and a final colour-cycling hue rotation (section 12); in the
**modes** style it is one motion mode per shape family. The steps:

1. **Builds a warp map for each displayed image's motion mode** (section 12): kaleidoscope fold, zoom, shockwave, shatter, spin
   (per-part rotation by radius with twist and ripple), sway, or liquid flow, plus the camera drift. One `cv2.remap` call per image
   then moves every pixel.
2. Blends the two current section images (each with its own mode during a crossfade). The **line-art** and **light-spot** layers are
   warped with the same map (so they stay attached to the artwork), with an extra outward push on drum hits.
3. Adjusts sharpness (unsharp mask), saturation and brightness.
4. Recolours the light spots (a new colour on every drum hit, stronger on hits).
5. Adds the glowing line art (accent colour of the image), particles and the onset flash.

Particles have fixed random starting positions (seeded) and drift outward over time, so the particle field is identical in every
version; only *how many* are visible depends on the controls. Frames are streamed straight to an H.264 file, and a small grayscale copy of
each frame is kept to measure **visual motion energy** in section 16.
""")

code(r"""
MAX_PARTICLES = VIS["particles_min"] + VIS["particles_range"]
_rng = np.random.default_rng(RANDOM_SEED)
PARTICLES = {                                     # fixed particle field shared by all renders
    "angle": _rng.uniform(0, 2 * np.pi, MAX_PARTICLES),
    "r0": _rng.uniform(0.05, 0.7, MAX_PARTICLES),
    "speed": _rng.uniform(0.02, 0.10, MAX_PARTICLES),     # fraction of frame per second
    "size": _rng.choice([1, 1, 2], MAX_PARTICLES),
}

def accent_color(img):
    # Bright colour derived from the image's average colour, so the line glow matches the artwork.
    c = img.reshape(-1, 3).mean(axis=0)
    return 0.55 * c / (c.max() + 1e-6) + 0.45

def hue_to_rgb(h):
    return np.array([0.5 + 0.5 * np.cos(2 * np.pi * (h - o)) for o in (0.0, 1 / 3, 2 / 3)], np.float32)

MOTION_MODES = {"star": "kaleidoscope", "tunnel": "zoom", "sunburst": "shockwave", "crystal": "shatter",
                "vortex": "spin", "flower": "sway", "waves": "flow"}
SECTION_MODES = [MOTION_MODES[f] for f in prompts.shape_family]     # one motion mode per section image
RADIAL_PARTICLES = {"kaleidoscope", "zoom", "shockwave", "shatter", "spin"}   # others: rising sparks
TILE_X, TILE_Y = (xx // (S / VIS["shatter_tiles"])).astype(int), (yy // (S / VIS["shatter_tiles"])).astype(int)

def polar_maps(r_src, theta_src):
    return CX + r_src * np.cos(theta_src) * (S / 2), CY + r_src * np.sin(theta_src) * (S / 2)

def warp_maps(p, mode, age, extra_scale=1.0):
    # Source sampling coordinates for every output pixel, for one motion mode.
    v = VIS
    if mode in ("spin", "sway"):
        # Per-part rotation by radius: core / inner star / outer star (+ twist and fine ripple)
        w_core, w_inner, w_outer = part_weights(RADIUS, v["core_radius"] * p.core_scale, v["inner_radius"])
        if mode == "spin":
            core_a, inner_a, outer_a, twist_k = p.core_angle, p.inner_angle, p.outer_angle, p.inner_twist
        else:   # sway: rock back and forth instead of spinning
            core_a, inner_a, outer_a, twist_k = 0.3 * p.sway_angle, p.sway_angle, -0.6 * p.sway_angle, 0.5 * p.inner_twist
        twist = twist_k * w_inner * (RADIUS - v["core_radius"])
        angle = np.deg2rad(w_core * core_a + w_inner * inner_a + w_outer * outer_a) + twist
        scale = (w_core * p.core_scale + w_inner + w_outer * p.outer_scale) * p.zoom * extra_scale
        r_src = RADIUS / scale + p.ripple_amp * np.sin(v["ripple_waves"] * 2 * np.pi * RADIUS - p.ripple_phase)
        mx, my = polar_maps(r_src, THETA - angle)
    elif mode == "kaleidoscope":
        # Fold the image into N mirrored wedges; the pattern slides through the wedges
        wedge = 2 * np.pi / N_POINTS
        folded = np.abs(np.mod(THETA - np.deg2rad(0.3 * p.inner_angle), wedge) - wedge / 2)
        r_src = RADIUS / (p.zoom * (1 + 0.5 * (p.core_scale - 1)) * extra_scale)
        mx, my = polar_maps(r_src, folded + np.deg2rad(p.outer_angle))
    elif mode == "zoom":
        # Fly into the image over the section, with a forward punch on every beat
        z = p.zoom * (1 + v["zoom_push"] * age) * (1 + 0.08 * p.beat_pulse) * extra_scale
        mx, my = polar_maps(RADIUS / z, THETA - np.deg2rad(0.2 * p.core_angle))
    elif mode == "shockwave":
        # A ring wave travels outward from the centre after every beat
        front = v["shock_speed"] * p.beat_age
        wave = v["shock_amp"] * np.exp(-((RADIUS - front) / v["shock_width"]) ** 2) * np.exp(-1.5 * p.beat_age)
        r_src = (RADIUS - wave) / (p.zoom * p.outer_scale * extra_scale)
        mx, my = polar_maps(r_src, THETA - np.deg2rad(0.15 * p.core_angle))
    elif mode == "shatter":
        # Tiles jump apart on each drum hit (new random offsets per hit) and snap back as the hit decays
        n = v["shatter_tiles"]
        offsets = np.random.default_rng(int(p.tile_seed)).uniform(-1, 1, (n, n, 2)) * v["shatter_amp"] * S * p.drum_pulse
        mx = CX + (xx - CX) / (p.zoom * extra_scale) + offsets[TILE_Y, TILE_X, 0]
        my = CY + (yy - CY) / (p.zoom * extra_scale) + offsets[TILE_Y, TILE_X, 1]
    else:   # "flow": liquid ripples across the whole frame
        amp = p.flow_amp * S
        mx = xx + amp * np.sin(2 * np.pi * yy / (0.6 * S) + p.flow_phase)
        my = yy + amp * np.cos(2 * np.pi * xx / (0.8 * S) + 0.7 * p.flow_phase) + 0.02 * S * np.sin(0.5 * p.flow_phase)
    # Slow camera drift for every mode, so nothing stays locked to the centre
    return (mx + p.drift_x * S).astype(np.float32), (my + p.drift_y * S).astype(np.float32)

def warp(img, maps):
    return cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def hue_rotate(img, turns):
    # Rotate colours around the grey axis (a hue shift that keeps brightness roughly constant).
    c, s_ = np.cos(2 * np.pi * turns), np.sin(2 * np.pi * turns)
    k, q = (1 - c) / 3, np.sqrt(1 / 3) * s_
    M = np.array([[c + k, k - q, k + q], [k + q, c + k, k - q], [k - q, k + q, c + k]], np.float32)
    return np.clip(img @ M.T, 0, 1)

R_MIRROR = 0.95                                     # radius at which the endless zoom mirrors back inward
ZONES = part_weights(RADIUS, VIS["core_radius"], VIS["inner_radius"])     # core / inner / outer rings of the frame
INSTR_RGB = {n: np.array(c, np.float32) for n, c in INSTRUMENT_COLORS.items()}

def kaleido_zoom_maps(p, scale, swirl, folds):
    # Kaleidoscope fold into `folds` segments + zoom by `scale` + radial mirroring (concentric repeats) + morph swirl.
    wedge = 2 * np.pi / folds
    folded = np.abs(np.mod(THETA - np.deg2rad(p.kaleido_angle), wedge) - wedge / 2)
    r = RADIUS / (scale * p.zoom) + p.ripple_amp * np.sin(VIS["ripple_waves"] * 2 * np.pi * RADIUS - p.ripple_phase)
    r = R_MIRROR - np.abs(np.mod(r, 2 * R_MIRROR) - R_MIRROR)         # triangle wave: mirror the radius
    theta = folded + np.deg2rad(p.outer_angle) + swirl * np.exp(-2 * RADIUS)
    mx, my = polar_maps(r, theta)
    return (mx + p.drift_x * S).astype(np.float32), (my + p.drift_y * S).astype(np.float32)

def layers_psychedelic(p):
    # Endless zoom: blend the image at zoom 2^f and 2^(f-1) (f = fractional zoom phase) -> seamless loop.
    a, b, t = int(p.img_a), int(p.img_b), p.blend
    f = p.zoom_phase % 1.0
    w_next = smoothstep(0.35, 0.65, f)          # one zoom level dominates; swap quickly mid-octave (less double exposure)
    swirl = VIS["morph_swirl"] * np.sin(np.pi * t) if t > 0 else 0.0
    shown = [a] + ([b] if t > 0 and b != a else [])
    out = {}
    for k in shown:
        img = np.zeros((S, S, 3), np.float32)
        lines = np.zeros((S, S), np.float32)
        lights = np.zeros((S, S), np.float32)
        folds = [(int(p.fold_a), 1 - p.fold_mix)] + ([(int(p.fold_b), p.fold_mix)] if p.fold_mix > 0 else [])
        for scale, w_zoom in ((2 ** f, 1 - w_next), (2 ** (f - 1), w_next)):
            for n_folds, w_fold in folds:                  # two segment counts are blended while they morph
                weight = w_zoom * w_fold
                if weight < 1e-3:
                    continue
                maps = kaleido_zoom_maps(p, scale, swirl, n_folds)
                img += weight * warp(images_for_render[k], maps)
                lines += weight * warp(layers[k]["lines"], maps)
                lights += weight * warp(layers[k]["lights"], maps)
        out[k] = (img, lines, lights)
    if len(shown) == 1:
        return out[a]
    # Morph: the new image appears where it is brightest first (luminance-keyed dissolve)
    img_b = out[b][0]
    mask = smoothstep(0.0, 1.0, 1.8 * t - 0.4 + 0.8 * (img_b.mean(axis=2) - 0.5))
    m3 = mask[..., None]
    return tuple(out[a][n] * (1 - (m3 if n == 0 else mask)) + out[b][n] * (m3 if n == 0 else mask) for n in range(3))

def layers_modes(p):
    # One motion mode per section image (section 12); crossfade between the two current images.
    a, b, t = int(p.img_a), int(p.img_b), p.blend
    shown = [(a, p.age_a, 1 - t), (b, p.age_b, t)] if t > 0 and b != a else [(a, p.age_a, 1.0)]
    img = np.zeros((S, S, 3), np.float32)
    lines = np.zeros((S, S), np.float32)
    lights = np.zeros((S, S), np.float32)
    for k, age, weight in shown:
        mode = SECTION_MODES[k]
        maps = warp_maps(p, mode, age)
        stretch = warp_maps(p, mode, age, extra_scale=p.lines_stretch)   # lines + lights push outward on drum hits
        img += weight * warp(images_for_render[k], maps)
        lines += weight * warp(layers[k]["lines"], stretch)
        lights += weight * warp(layers[k]["lights"], stretch)
    return img, lines, lights

def render_frame(p, frame_idx, images):
    global images_for_render
    images_for_render = images
    a, b, t = int(p.img_a), int(p.img_b), p.blend
    psychedelic = VISUAL_STYLE == "psychedelic"

    # 1-2. Move the image parts (style-specific), including the line-art and light-spot layers
    img, lines, lights = layers_psychedelic(p) if psychedelic else layers_modes(p)

    # 3. Sharpness (centroid), saturation (RMS, + beat pop in psychedelic style) and brightness (RMS)
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = img + (p.sharpness - 1.0) * 2.0 * (img - blur)
    gray = img.mean(axis=2, keepdims=True)
    sat = p.saturation * (VIS["psy_saturation"] if psychedelic else 1.0) + \
          (VIS["beat_saturation"] * p.beat_pulse if psychedelic else 0.0)
    img = np.clip((gray + sat * (img - gray)) * p.brightness, 0, 1)
    if psychedelic:                                              # contrast curve around mid-grey
        img = np.clip(0.5 + VIS["psy_contrast"] * (img - 0.5), 0, 1)

    # 4. Light spots take on a colour that changes on every drum hit (they glow brighter on hits)
    g = np.clip(lights * p.lights_gain, 0, 1)[..., None]
    img = img * (1 - g) + g * hue_to_rgb(p.light_hue)
    # 5. Glowing line art (screen blend), colour cycling (psychedelic), particles, onset flash
    glow = lines + 0.8 * cv2.GaussianBlur(lines, (0, 0), 4.0)
    if psychedelic:
        img = hue_rotate(img, p.hue_shift)
        # Instruments inside the design: each ring's lines glow in its instrument's colour, as loud as the instrument,
        # and the ring blooms when the instrument enters (colours applied after hue cycling so they stay recognisable)
        w_core, w_inner, w_outer = ZONES
        ring_rgb = (w_core[..., None] * INSTR_RGB["bass"] * (0.15 + p.lvl_bass)
                    + w_inner[..., None] * INSTR_RGB["vocals"] * (0.15 + p.lvl_lead)
                    + w_outer[..., None] * INSTR_RGB["other"] * (0.15 + p.lvl_other))
        line_rgb = glow[..., None] * ring_rgb * VIS["instrument_line_gain"] * (0.6 + p.lines_gain)
        bloom = VIS["instrument_entry_glow"] * (w_core[..., None] * INSTR_RGB["bass"] * p.enter_bass
                                                + w_inner[..., None] * INSTR_RGB["vocals"] * p.enter_vocals
                                                + w_outer[..., None] * INSTR_RGB["other"] * p.enter_other
                                                + lights[..., None] * INSTR_RGB["drums"] * 2 * p.enter_drums)
        img = 1 - (1 - img) * (1 - np.clip(line_rgb + bloom, 0, 1))
    else:
        line_col = accent_color(images[a] if t < 0.5 else images[b])
        img = 1 - (1 - img) * (1 - np.clip(glow[..., None] * line_col * p.lines_gain, 0, 1))
    overlay = np.zeros((S, S, 3), np.float32)
    n = int(p.n_particles)
    jit = np.random.default_rng(frame_idx).normal(0, 1.5 * p.jitter, (n, 2))
    if psychedelic or SECTION_MODES[a if t < 0.5 else b] in RADIAL_PARTICLES:   # burst outward from the centre
        r = (PARTICLES["r0"][:n] + PARTICLES["speed"][:n] * p.time_sec) % 0.75 * S
        px = S / 2 + r * np.cos(PARTICLES["angle"][:n]) + jit[:, 0]
        py = S / 2 + r * np.sin(PARTICLES["angle"][:n]) + jit[:, 1]
    else:                                                                      # rise like sparks
        x0 = PARTICLES["angle"][:n] / (2 * np.pi) * S
        px = x0 + 0.02 * S * np.sin(p.time_sec * 1.5 + x0) + jit[:, 0]
        py = S - (PARTICLES["r0"][:n] * S + PARTICLES["speed"][:n] * 2 * S * p.time_sec) % S + jit[:, 1]
    for x0, y0, sz in zip(px.astype(int), py.astype(int), PARTICLES["size"][:n]):
        cv2.circle(overlay, (int(x0), int(y0)), int(sz), (1.0, 1.0, 1.0), -1, cv2.LINE_AA)
    img = 1 - (1 - img) * (1 - overlay) + p.flash
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def mux_audio(video_path, audio_path, out_path):
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(video_path), "-i", str(audio_path),
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
           "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",   # standard stereo audio plays everywhere
           "-shortest", "-movflags", "+faststart", str(out_path)]
    subprocess.run(cmd, check=True)
    Path(video_path).unlink()            # remove the silent intermediate so only the video WITH sound remains
    return out_path

def media_info(path):
    # Returns (video seconds, has audio stream) using the bundled ffmpeg.
    n_frames, secs = imageio_ffmpeg.count_frames_and_secs(str(path))
    probe = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    return n_frames, secs, "Audio:" in probe


def render_video(params, path, images, seconds=None):
    # Render frames (by timestamp) to a silent H.264 MP4; return per-frame motion energy and mean luminance.
    rows = params if seconds is None else params[params.time_sec < seconds]
    writer = imageio_ffmpeg.write_frames(str(path), (OUTPUT_SIZE, OUTPUT_SIZE), fps=FPS, codec="libx264",
                                         pix_fmt_out="yuv420p", quality=5, macro_block_size=1)
    writer.send(None)
    prev, motion, lum = None, [], []
    for i, p in enumerate(rows.itertuples(index=False)):
        frame = render_frame(p, i, images)
        writer.send(np.ascontiguousarray(frame))
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (128, 128), interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32) / 255.0
        motion.append(0.0 if prev is None else float(np.mean(np.abs(small - prev))))
        lum.append(float(small.mean()))
        prev = small
    writer.close()
    return pd.DataFrame({"time_sec": rows.time_sec.values, "motion": motion, "luminance": lum})
""")

md(r"""
### 13.1 Quick preview (first `PREVIEW_SECONDS` seconds, with sound)
A short render to check the look before committing to the full video.
""")

code(r"""
preview_path = OUTPUT_DIR / "preview.mp4"
_ = render_video(audio_params, OUTPUT_DIR / "preview_silent.mp4", section_images, seconds=PREVIEW_SECONDS)
mux_audio(OUTPUT_DIR / "preview_silent.mp4", AUDIO_WAV, preview_path)

# Contact sheet: one frame every second of the preview
sample_frames = [render_frame(p, i, section_images)
                 for i, p in enumerate(audio_params.itertuples(index=False)) if i % FPS == 0 and i < PREVIEW_SECONDS * FPS]
fig, axes = plt.subplots(1, len(sample_frames), figsize=(2.2 * len(sample_frames), 2.5))
for k, (ax, f) in enumerate(zip(axes, sample_frames)):
    ax.imshow(f); ax.axis("off"); ax.set_title(f"t = {k} s", fontsize=9)
plt.tight_layout(); plt.show()
display(Video(str(preview_path), width=320))
""")

md(r"""
### 13.2 Full-length render (silent)
""")

code(r"""
audio_silent = OUTPUT_DIR / "visualizer_silent.mp4"
audio_render = render_video(audio_params, audio_silent, section_images)
print(f"Rendered {len(audio_render)} frames -> {audio_silent}")
""")

# =============================================================================
# 14. Mux
# =============================================================================
md(r"""
## 14. Mux the original audio into the final MP4

The silent video and `outputs/audio_clip.wav` are combined with the FFmpeg CLI. The video stream is copied unchanged (`-c:v copy`) and the
audio is encoded as 44.1 kHz stereo AAC. The silent intermediate file is then deleted, so **`outputs/visualizer_audio.mp4` is the final
video with the song**. Durations are checked to rule out audio/video drift.
""")

code(r"""
FINAL_VIDEO = mux_audio(audio_silent, AUDIO_WAV, OUTPUT_DIR / "visualizer_audio.mp4")
n_frames_out, secs_out, has_audio = media_info(FINAL_VIDEO)
print(f"{FINAL_VIDEO}: {n_frames_out} frames, {secs_out:.2f} s, audio stream: {has_audio}")
assert has_audio and n_frames_out == N_FRAMES and abs(secs_out - DURATION) < 1 / FPS + 0.05

display(Video(str(FINAL_VIDEO), width=512))
""")

# =============================================================================
# 15. Baseline
# =============================================================================
md(r"""
## 15. Render the non-audio-conditioned baseline

The baseline uses the **same song, the same generated images, the same renderer, effect types, length and frame rate**. The only
difference is where the control signals come from: preset procedural rules that **never read any LibROSA feature**.

| Control | Audio-conditioned | Baseline (preset) |
|---|---|---|
| bass, mid, high, centroid, instrument stems, lead | measured, normalized | constant `BASELINE_LEVEL` = 0.5 (mid-range of every effect) |
| beat pulse, drum-hit pulse | decays from each detected beat / drum hit | same decay shape, repeating every `BASELINE_PULSE_PERIOD` = 0.70 s (not the song's tempo) |
| light colour changes | on every drum hit | on every preset pulse |
| onset flash | onset events × strength | fixed flash every `BASELINE_FLASH_PERIOD` = 2.8 s at half strength |
| brightness | RMS | slow sine cycle (period 10 s) around the mid level |
| section transitions | detected boundaries; 0.25 s cut + flash at dramatic beat changes | equal time windows, always the 1 s crossfade |

The baseline is meant to look polished, not deliberately bad: it simply removes the audio-conditioning mechanism.
""")

code(r"""
def baseline_controls(times):
    # Preset control signals with the same columns as the audio timeline, built ONLY from time.
    c = pd.DataFrame({"time_sec": times})
    for name in ["bass", "mid", "high", "centroid", "drums", "bass_stem", "other", "vocals", "lead"]:
        c[name] = BASELINE_LEVEL
    c["rms"] = BASELINE_LEVEL + 0.3 * np.sin(2 * np.pi * times / BASELINE_BRIGHT_PERIOD)
    c["beat_pulse"] = np.exp(-PULSE_DECAY * (times % BASELINE_PULSE_PERIOD))
    c["drum_pulse"] = c["beat_pulse"]
    c["drum_hits"] = np.floor(times / BASELINE_PULSE_PERIOD)
    c["drum_unit"] = c["beat_pulse"]
    c["active_count"] = 3                                   # constant: no instrument changes in the baseline
    for n in ["bass", "vocals", "other", "drums"]:
        c[f"enter_{n}"] = 0.0
    c["onset_pulse"] = 0.5 * np.exp(-PULSE_DECAY * (times % BASELINE_FLASH_PERIOD))
    return c

baseline_edges = np.linspace(0, DURATION, n_sections + 1)          # fixed transition times
baseline_params = visual_params(baseline_controls(frame_times), baseline_edges)

baseline_silent = OUTPUT_DIR / "baseline_silent.mp4"
baseline_render = render_video(baseline_params, baseline_silent, section_images)
BASELINE_VIDEO = mux_audio(baseline_silent, AUDIO_WAV, OUTPUT_DIR / "baseline_audio.mp4")
n_b, secs_b, audio_b = media_info(BASELINE_VIDEO)
print(f"{BASELINE_VIDEO}: {n_b} frames, {secs_b:.2f} s, audio stream: {audio_b}")
assert audio_b and n_b == N_FRAMES

display(Video(str(BASELINE_VIDEO), width=512))
""")

md(r"""
### 15.1 Optional robustness check: time-shifted controls
Circularly shifting the real control timeline by `BASELINE_SHIFT_SECONDS` keeps the exact statistics of the audio controls but moves
every event to the wrong moment. Set `RENDER_SHIFTED_CHECK = True` to render it. This is a secondary check, not the primary baseline.
""")

code(r"""
shifted_render = None
if RENDER_SHIFTED_CHECK:
    shift = int(BASELINE_SHIFT_SECONDS * FPS)
    shifted_ctrl = timeline.copy()
    cols = ["rms", "bass", "mid", "high", "centroid", "beat_pulse", "onset_pulse",
            "drums", "bass_stem", "other", "vocals", "lead", "drum_pulse", "drum_hits", "drum_unit",
            "active_count", "enter_bass", "enter_vocals", "enter_other", "enter_drums"]
    shifted_ctrl[cols] = np.roll(timeline[cols].values, shift, axis=0)
    shifted_params = visual_params(shifted_ctrl, section_edges)
    shifted_render = render_video(shifted_params, OUTPUT_DIR / "shifted_silent.mp4", section_images)
    mux_audio(OUTPUT_DIR / "shifted_silent.mp4", AUDIO_WAV, OUTPUT_DIR / "shifted_audio.mp4")
    print("Rendered time-shifted check video")
else:
    print("Skipped (RENDER_SHIFTED_CHECK = False)")
""")

# =============================================================================
# 16. Technical checks
# =============================================================================
md(r"""
## 16. Technical alignment checks

These are **implementation / alignment checks**, not proof that one video is better art. They are measured on the **rendered frames**,
not on the control parameters (e.g. `corr(RMS, brightness parameter)` would be ~1 by construction and says nothing).

* **Visual motion energy:** `motion[t] = mean(|gray[t] − gray[t−1]|)` on 128×128 grayscale frames.
* **Correlation** (Pearson and Spearman) of motion energy with RMS, onset strength and the drum / bass stems, for both videos.
* **Cross-correlation lag:** the lag at which onset strength and motion energy line up best. For a well-synchronized visualizer the
  peak should be near **0 ms**.
* **QA:** beats map to pulses, crossfades sit on section boundaries, video length = audio length.

Section crossfades are excluded from the correlations (they are large, slow image changes that would dominate motion energy in both videos).
""")

code(r"""
MAX_LAG_FRAMES = FPS                                  # search lags within +/- 1 s

def cross_corr_lag(a, b, max_lag=MAX_LAG_FRAMES):
    # Lag (frames) maximizing corr(a[t], b[t + lag]); positive = visuals respond after the audio.
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    lags = np.arange(-max_lag, max_lag + 1)
    cc = np.array([np.mean(a[max(0, -l):len(a) - max(0, l)] * b[max(0, l):len(b) - max(0, -l)]) for l in lags])
    return lags, cc

def not_in_crossfade(params):
    return params.blend.values == 0

versions = {"audio-conditioned": (audio_render, audio_params), "baseline": (baseline_render, baseline_params)}
if shifted_render is not None:
    versions["time-shifted"] = (shifted_render, audio_params)

metric_rows, lag_curves = [], {}
for name, (rend, params) in versions.items():
    keep = not_in_crossfade(params) & (np.arange(N_FRAMES) > 0)      # drop crossfades and the first frame
    m = rend.motion.values
    lags, cc = cross_corr_lag(timeline.onset.values[keep], m[keep])
    lag_curves[name] = (lags, cc)
    for feature in ["rms", "onset", "drums", "bass_stem"]:
        x = timeline[feature].values
        if np.std(x[keep]) == 0:                                     # absent stem: correlation undefined
            continue
        metric_rows.append({"version": name, "feature": feature,
                            "pearson_r": scipy.stats.pearsonr(x[keep], m[keep])[0],
                            "spearman_rho": scipy.stats.spearmanr(x[keep], m[keep])[0],
                            "best_lag_ms": lags[np.argmax(cc)] / FPS * 1000 if feature == "onset" else np.nan})

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
metrics.round(3)
""")

code(r"""
fig, axes = plt.subplots(3, 1, figsize=(12, 9))
ax = axes[0]
ax.plot(frame_times, robust_normalize(audio_render.motion.values), color=COLORS["rms"], lw=1, label="audio-conditioned motion")
ax.plot(frame_times, timeline.onset, color=COLORS["onset"], lw=0.8, alpha=0.7, label="onset strength")
ax.set(title="Audio-conditioned: visual motion energy vs onset strength", ylabel="normalized", xlim=(0, DURATION))
ax.legend(loc="upper left")
ax = axes[1]
ax.plot(frame_times, robust_normalize(baseline_render.motion.values), color=COLORS["baseline"], lw=1, label="baseline motion")
ax.plot(frame_times, timeline.onset, color=COLORS["onset"], lw=0.8, alpha=0.7, label="onset strength")
ax.set(title="Baseline: visual motion energy vs onset strength", ylabel="normalized", xlabel="time (s)", xlim=(0, DURATION))
ax.legend(loc="upper left")
ax = axes[2]
for name, color in [("audio-conditioned", COLORS["rms"]), ("baseline", COLORS["baseline"]), ("time-shifted", COLORS["high"])]:
    if name in lag_curves:
        lags, cc = lag_curves[name]
        ax.plot(lags / FPS * 1000, cc, color=color, lw=1.8, label=name)
ax.axvline(0, color="#52514e", lw=0.8, ls="--")
ax.set(title="Cross-correlation: onset strength vs visual motion", xlabel="lag (ms), positive = visuals after audio",
       ylabel="correlation")
ax.legend(loc="upper right")
plt.tight_layout(); plt.savefig(FIG_DIR / "alignment.png"); plt.show()
""")

code(r"""
# ---- Control-timeline and output QA -------------------------------------------------
qa = {}
beat_idx = np.clip(np.ceil(beat_times * FPS).astype(int), 0, N_FRAMES - 1)   # first frame at/after each beat
qa["mean beat_pulse at beat frames (should be close to 1)"] = round(float(timeline.beat_pulse.values[beat_idx].mean()), 3)

fade_mid = [audio_params.time_sec[(audio_params.img_a == k) & (audio_params.img_b == k + 1)].mean()
            for k in range(n_sections - 1)]
qa["crossfade centres (s)"] = np.round(fade_mid, 2).tolist()
qa["section boundaries (s)"] = np.round(section_edges[1:-1], 2).tolist()
qa["max crossfade offset from boundary (s)"] = round(float(np.max(np.abs(np.array(fade_mid) - section_edges[1:-1]))), 3)
qa["audio duration (s)"] = round(DURATION, 3)
qa["final video duration (s)"] = round(secs_out, 3)
qa["baseline video duration (s)"] = round(secs_b, 3)
qa["final video has audio"] = has_audio
for k, v in qa.items():
    print(f"{k:<55} {v}")
assert qa["max crossfade offset from boundary (s)"] <= 1 / FPS + 1e-6
""")

# =============================================================================
# 17. Survey
# =============================================================================
md(r"""
## 17. Human evaluation: survey design

The survey is the **main effectiveness measure**; the technical checks above only confirm the system behaves as intended.

**Design:** within-subjects A/B comparison. Each participant watches both videos of the same clip (with sound):
the audio-conditioned visualizer and the non-audio-conditioned baseline. Labels are neutral ("Video A / Video B") and the order is
**counterbalanced**: half of the participants see the audio-conditioned version first.

**Ratings (1 = not at all, 5 = very much), asked for each video:**
1. How synchronized do the visuals feel with the music? *(primary outcome)*
2. How well do the visuals reflect changes in the music?
3. How connected do the visuals feel to the audio?
4. How visually appealing is the video?

**Final question:** Which version better represents the music overall? (A / B)

**Analysis:** means and medians per question, paired differences (audio − baseline), preference percentage, and optionally a paired
Wilcoxon signed-rank test. With a small convenience sample the results describe *this group*, not people in general.
Confirm participant / consent rules with the instructor before collecting responses.

The cell below writes an **empty response template** and analyzes `outputs/survey_responses.csv` once real responses are entered.
No survey results are invented here.
""")

code(r"""
SURVEY_COLS = ["participant", "order", "version", "q1_sync", "q2_changes", "q3_connection", "q4_appeal", "preferred_version"]
template = OUTPUT_DIR / "survey_template.csv"
pd.DataFrame(columns=SURVEY_COLS).to_csv(template, index=False)
print(f"Template written: {template}")
print("One row per participant x version; order = 'audio_first' or 'baseline_first'; "
      "version / preferred_version = 'audio' or 'baseline'.")

def analyze_survey(df):
    questions = ["q1_sync", "q2_changes", "q3_connection", "q4_appeal"]
    summary = df.groupby("version")[questions].agg(["mean", "median"]).round(2)
    wide = df.pivot(index="participant", columns="version", values=questions)
    rows = []
    for q in questions:
        diff = wide[(q, "audio")] - wide[(q, "baseline")]
        p = scipy.stats.wilcoxon(diff).pvalue if (diff != 0).any() else np.nan
        rows.append({"question": q, "mean_diff_audio_minus_baseline": diff.mean(), "wilcoxon_p": p, "n": len(diff)})
    pref = df.drop_duplicates("participant").preferred_version.value_counts(normalize=True) * 100
    return summary, pd.DataFrame(rows).round(3), pref.round(1)

responses = OUTPUT_DIR / "survey_responses.csv"
if responses.exists() and len(pd.read_csv(responses)) > 0:
    summary, paired, pref = analyze_survey(pd.read_csv(responses))
    display(summary, paired)
    print("Overall preference (%):", pref.to_dict())
else:
    print("No survey responses yet - copy the template to outputs/survey_responses.csv and fill it in.")
""")

# =============================================================================
# 18-20. Summary, limitations, future work
# =============================================================================
md(r"""
## 18. Results summary
""")

code(r"""
a = metrics[(metrics.version == "audio-conditioned")].set_index("feature")
b = metrics[(metrics.version == "baseline")].set_index("feature")
print(f"Audio source: {AUDIO_SOURCE} | {DURATION:.1f} s | tempo estimate {tempo:.1f} BPM")
print(f"Sections: {n_sections} ({SECTION_METHOD}) at {np.round(section_edges[1:-1], 2).tolist()} s")
print(f"Stems: {STEM_METHOD} | present: {[n for n in STEM_NAMES if stem_present[n]]} | drum hits: {len(drum_times)}")
print(f"Vibe: {VIBE} ({VIBE_METHOD}) | key {KEY} | lyrics: {LYRICS_NOTE}")
print(f"Shapes per section: {prompts.shape_family.tolist()} | dramatic beat changes at {DRAMATIC_BOUNDARIES.tolist()} s")
print(f"Visual style: {VISUAL_STYLE}" + ("" if VISUAL_STYLE == "psychedelic" else f" | motion per section: {SECTION_MODES}"))
print(f"Materials: {MATERIAL_TEXT}")
print(f"Instrument changes: {len(instrument_events)} " +
      ", ".join(f"{r.instrument} {r.change} {r.time_sec:.1f}s" for r in instrument_events.itertuples()))
print(f"Images: {prompts.generated_by.unique().tolist()} | seeds {prompts.seed.tolist()} | {GEN_STEPS} steps\n")
print("Motion energy vs audio (rendered frames, crossfades excluded):")
for f in ["rms", "onset"]:
    print(f"  {f:>5}: audio-conditioned r = {a.loc[f, 'pearson_r']:+.2f} | baseline r = {b.loc[f, 'pearson_r']:+.2f}")
print(f"  best onset->motion lag: audio-conditioned {a.loc['onset', 'best_lag_ms']:+.0f} ms | "
      f"baseline {b.loc['onset', 'best_lag_ms']:+.0f} ms")
print(f"\nOutputs: {FINAL_VIDEO}, {BASELINE_VIDEO}")
""")

md(r"""
**Confound to keep in mind:** the section images were generated from section statistics, so loud sections get busier artwork, and
busier artwork produces more frame-to-frame change even under the baseline's preset motion. Correlations with **slow** features (RMS,
stem levels) are therefore partly driven by *which image is on screen* in both videos. Correlation with **onset strength** (fast,
moment-to-moment events) is the cleaner test of synchronization.

**How to read this:** with strongly periodic music (the demo has 16th-note hi-hats every 125 ms) the cross-correlation curve
is itself periodic, so the useful signal is that the *highest* peak sits at 0 ms and is clearly above the baseline curve. If the audio-conditioned version has clearly higher correlations and a cross-correlation peak near 0 ms while the
baseline stays near zero, the *implementation* is aligned with the audio as designed. Whether viewers actually **perceive** better
synchronization is answered by the survey (section 17), not by these numbers.

## 19. Limitations

* **Features are proxies.** RMS is a short-time energy/amplitude proxy, not perceived loudness; spectral centroid is a brightness proxy,
  not pitch; beat tracking is an estimate and can lock onto half/double tempo.
* **Band limits and thresholds are design choices** (20–250 / 250–2000 / 2000–8000 Hz; low/medium/high at 1/3 and 2/3) and are not
  universal perceptual boundaries.
* **Normalization is relative to the clip**, so "high bass" means high *for this song*, not in absolute terms.
* **Sections are change regions**, not musical verses, choruses or drops.
* **Wavelength-inspired mapping:** large forms for bass and fine detail for highs is a cross-modal design metaphor; the visuals do not
  physically represent acoustic wavelength.
* **Stems are estimates.** Demucs separation leaks between stems, and "other" mixes every remaining instrument.
* **The vibe classifier is not trained or validated.** The acoustic prototypes are hand-written design rules, CLAP is zero-shot, and
  the feature scalings (e.g. 80–125 BPM) are assumptions. Beat trackers can report half or double tempo, and loudness depends on
  mastering. Close scores mean the choice is nearly arbitrary, which is why `VIBE_OVERRIDE` exists.
* **Lyrics** come from automatic transcription of a separated vocal stem and can be wrong; binary sentiment misses irony and nuance.
* **Dramatic changes are only checked at detected section boundaries**, with hand-set weights and threshold; a beat change in the
  middle of a section is not caught, and local tempo estimates over 3 s windows are noisy (hence their low weight).
* **Shape rules are design metaphors** (tempo → number of points, minor key → shards); the key estimate can be wrong for songs with
  modulations or ambiguous harmony.
* **Parts are found by radius**, which works because the prompts ask for a centred radial composition; arbitrary scenes
  (e.g. a crowd) would need a segmentation model, and people are not animated.
* **Prompt following is limited.** SD-Turbo is a small, few-step model; images reflect the descriptors only loosely, and the 77-token
  CLIP limit constrains prompt detail.
* **Technical metrics test the implementation, not perception.** Motion energy is one crude output measure.
* **Small survey samples** do not generalize beyond the participants.
* The development clip is a **synthesized** track with very clear sections; real songs will be less tidy.

## 20. Future work

* Run the identical notebook on one or two **real songs** (set `AUDIO_PATH`) to check that the pipeline is not clip-specific.
* **Train a vibe classifier** (CNN on mel-spectrograms, or a small head on CLAP / YAMNet embeddings) once a labelled set of songs
  exists, and validate it against listener ratings; use an LSTM / per-section vibe to let the theme evolve through the song.
* Replace radial parts with **segmentation masks** (e.g. Segment Anything) to animate arbitrary objects such as lights or silhouettes
  in a themed scene.
* Try a stronger model (SDXL-Turbo) if GPU memory allows, and compare prompt adherence.
* Add the optional **time-shifted** comparison as a second baseline and an accessibility-focused evaluation.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "genai_audio_visualizer.ipynb")
print(f"Wrote genai_audio_visualizer.ipynb with {len(cells)} cells")
