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
| **Procedural** | Python (NumPy + OpenCV) animates and crossfades those images frame by frame, driven by the real audio features: beats, low/mid/high frequency energy, RMS energy, onset strength and spectral brightness. |

### Research question
> Can audio-derived features be used to guide generative artwork and procedural animation to create a visualization that feels more synchronized and connected to the music than a non-audio-conditioned visualizer?

### Hypothesis
> The audio-conditioned visualizer will receive higher ratings for synchronization and perceived connection to the music than the non-audio-conditioned baseline.

To test this, the notebook renders **two videos** that share the same song, the same generated images and the same effect types:

1. **Audio-conditioned visualizer**: motion is driven by the extracted audio features.
2. **Baseline**: the same effects are driven by preset, fixed-rate rules that never look at the audio.

### What this notebook deliberately does *not* do
No frame-by-frame diffusion video, no real-time rendering, no model training, no mood classifier, no 3D. Images are generated once (3–4 total) and cached.
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
          "centroid": "#e87ba4", "onset": "#6250d6", "beat": "#52514e", "baseline": "#e34948"}
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

# ---- Sections -------------------------------------------------------------------
N_SECTIONS      = 4           # target number of analysis sections (3-4)
MIN_SECTION_SEC = 4.5         # minimum section length; also the minimum spacing between boundaries
ACTIVITY_WEIGHTS = {"rms": 0.50, "onset": 0.30, "bass": 0.20}
ACTIVITY_SMOOTH_SEC = 1.0     # extra smoothing before looking for change points

# ---- Image generation ---------------------------------------------------------------
MODEL_ID        = "stabilityai/sd-turbo"
IMAGE_SIZE      = 512         # SD-Turbo's native resolution
GEN_STEPS       = 2           # SD-Turbo is designed for 1-4 steps
RANDOM_SEED     = 42          # same seed for every section -> related compositions
FORCE_REGENERATE = False      # True = ignore cached images and generate again

# ---- Video ----------------------------------------------------------------------------
FPS               = 24
OUTPUT_SIZE       = 512       # square video, pixels
CROSSFADE_SECONDS = 1.0       # length of the blend between section images
PREVIEW_SECONDS   = 8.0       # length of the quick preview render

# ---- Visual mapping coefficients (section 12) ------------------------------------------
VIS = {
    "zoom_base": 1.02,  "zoom_bass": 0.10,  "zoom_beat": 0.03,   # image scale
    "bg_rot_base": 2.0,  "bg_rot_mid": 6.0,                     # background rotation, deg/s
    "ring_base": 0.22,   "ring_beat": 0.18,                     # ring radius (fraction of frame) and beat expansion
    "ring_thick_base": 2, "ring_thick_bass": 12,                # ring thickness (px)
    "poly_rot_base": 15.0, "poly_rot_mid": 90.0,                # polygon rotation, deg/s
    "ribbon_amp_base": 0.02, "ribbon_amp_mid": 0.10,            # ribbon amplitude (fraction of frame)
    "particles_min": 40,  "particles_range": 360,              # number of visible particles
    "brightness_base": 0.75, "brightness_rms": 0.40,           # overall gain
    "saturation_base": 0.80, "saturation_rms": 0.50,
    "sharpness_base": 0.90, "sharpness_centroid": 0.30,        # >1 sharpens, <1 softens
    "flash_onset": 0.30,                                       # max additive white flash
    "overlay_strength": 0.85,                                  # opacity of the procedural layers
}

# ---- Baseline (section 15): fixed rules that never read the audio ------------------------
BASELINE_PULSE_PERIOD  = 0.70   # seconds between preset pulses (deliberately not the song's tempo)
BASELINE_FLASH_PERIOD  = 2.80   # seconds between preset flashes
BASELINE_BRIGHT_PERIOD = 10.0   # seconds per slow brightness cycle
BASELINE_LEVEL         = 0.5    # fixed "medium" value used in place of bass/mid/high/rms/centroid

RENDER_SHIFTED_CHECK   = False  # optional robustness check: audio controls circularly shifted in time
BASELINE_SHIFT_SECONDS = 6.0

# ---- Derived paths --------------------------------------------------------------------------
IMAGE_DIR = OUTPUT_DIR / "images"
FIG_DIR   = OUTPUT_DIR / "figures"
for d in (OUTPUT_DIR, IMAGE_DIR, FIG_DIR):
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
timeline["section_id"] = 0          # placeholder, set in section 8

# Acceptance checks
assert len(timeline) == N_FRAMES == round(DURATION * FPS)
assert not timeline.isna().any().any(), "NaNs in timeline"
controls = ["rms", "bass", "mid", "high", "centroid", "onset", "beat_pulse", "onset_pulse"]
assert timeline[controls].apply(lambda c: c.between(0, 1).all()).all(), "controls out of [0, 1]"
print(f"Timeline: {N_FRAMES} frames = {DURATION:.2f} s x {FPS} fps, all controls within [0, 1], no NaNs")
timeline.head(8).round(3)
""")

code(r"""
fig, ax = plt.subplots(figsize=(12, 2.8))
seg = timeline[(timeline.time_sec >= 8) & (timeline.time_sec < 14)]
ax.plot(seg.time_sec, seg.beat_pulse, color=COLORS["beat"], lw=1.5, label="beat pulse envelope")
ax.plot(seg.time_sec, seg.bass, color=COLORS["bass"], lw=1.5, label="bass (smoothed)")
ax.vlines(beat_times[(beat_times >= 8) & (beat_times < 14)], 0, 1, color=COLORS["beat"], lw=0.6, ls=":")
ax.set(title="Control timeline close-up (8-14 s): each detected beat (dotted) becomes a decaying pulse",
       xlabel="time (s)", ylabel="value (0-1)")
ax.legend(loc="upper right")
plt.tight_layout(); plt.savefig(FIG_DIR / "timeline_closeup.png"); plt.show()
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
# 10. Prompts
# =============================================================================
md(r"""
## 10. Build GenAI prompts from section features

The diffusion model never sees the waveform. Instead each section goes through a transparent chain:

`section statistics → low / medium / high levels → visual descriptors → text prompt → image`

**Levels** (documented design thresholds): `0.00–0.33 → low`, `0.33–0.67 → medium`, `0.67–1.00 → high`.

**Wavelength-inspired visual grammar.** Low-frequency sound has longer physical wavelengths and high-frequency sound shorter ones, so the
mapping uses *large, broad* forms for bass and *small, fine* details for highs. This is a cross-modal **design choice**; the program
measures frequency, and the images do not physically depict acoustic wavelength.

**Prompt length.** SD-Turbo's CLIP text encoder reads at most **77 tokens**; anything beyond is silently cut off. So the model prompt is
kept compact (shared base style + one short phrase per feature), and the full readable sentence is stored separately in
`section_description`. A token count check below confirms nothing is truncated. (With `guidance_scale = 0`, SD-Turbo ignores negative
prompts, so "no text / no people" is enforced by keeping the vocabulary abstract rather than by a negative prompt.)
""")

code(r"""
def level(x):
    return "low" if x < 1 / 3 else ("medium" if x < 2 / 3 else "high")

# Shared visual identity: identical start for every prompt so the images feel related
BASE_STYLE = ("abstract geometric audio-visual artwork, luminous digital art, centered composition, "
              "layered geometry, atmospheric depth, dark background")

# One short descriptor per feature and level (guide section 9.3)
DESCRIPTORS = {
    "rms":      {"low": "sparse calm dim open space",
                 "medium": "layered active moderately bright",
                 "high": "dense intense dramatic glow"},
    "bass":     {"low": "small delicate central form",
                 "medium": "medium circular structures",
                 "high": "huge circular rings, heavy central mass, broad waves"},
    "mid":      {"low": "simple inner structure",
                 "medium": "some layered polygons",
                 "high": "layered rotating polygons and ribbons"},
    "high":     {"low": "smooth surfaces",
                 "medium": "some fine lines",
                 "high": "fine particles, thin lines, sparkling micro detail"},
    "centroid": {"low": "soft rounded edges, warm amber violet light",
                 "medium": "balanced edges, violet and teal light",
                 "high": "crisp sharp edges, cool cyan white highlights"},
    "onset_density_rel": {"low": "few separate elements",
                          "medium": "several accents",
                          "high": "fragmented geometry, many bursts"},
}
FEATURES_FOR_PROMPT = ["rms", "bass", "mid", "high", "centroid", "onset_density_rel"]

def build_prompt(row):
    levels = {f: level(row[f]) for f in FEATURES_FOR_PROMPT}
    phrases = [DESCRIPTORS[f][levels[f]] for f in FEATURES_FOR_PROMPT]
    prompt = BASE_STYLE + ", " + ", ".join(phrases)
    description = (f"This section has {levels['rms']} energy with {levels['bass']} low-frequency presence, "
                   f"{levels['mid']} mid-frequency activity, {levels['high']} high-frequency detail, "
                   f"{levels['centroid']} spectral brightness and {levels['onset_density_rel']} onset density.")
    return prompt, description, levels

records = []
for _, row in sections.iterrows():
    prompt, description, levels = build_prompt(row)
    s = int(row.section)
    records.append({"section": s, **{f"{k}_level": v for k, v in levels.items()},
                    "section_description": description, "prompt": prompt,
                    "seed": RANDOM_SEED, "model": MODEL_ID, "steps": GEN_STEPS, "guidance_scale": 0.0,
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

prompts[["section", "rms_level", "bass_level", "mid_level", "high_level", "centroid_level", "onset_density_rel_level"]]
""")

# =============================================================================
# 11. Generate images
# =============================================================================
md(r"""
## 11. Generate (or load cached) section images

Model: **[SD-Turbo](https://huggingface.co/stabilityai/sd-turbo)** (distilled from Stable Diffusion 2.1, built for 1–4 steps at 512×512).
It is light enough for a free Colab T4 or an Apple-silicon laptop. Before publishing results, check the model card's license terms
(SD-Turbo is released under the Stability AI community license).

**Caching:** each image is saved to `outputs/images/section_N.png` together with `outputs/prompts.csv`. On later runs an image is only
regenerated if its prompt or settings changed (or `FORCE_REGENERATE = True`), so the notebook can be re-run without the GPU.

**Consistency:** all sections use the same seed, so the starting noise is identical and the compositions stay related; only the prompt changes.

If diffusion is unavailable (no `diffusers` / download fails), a clearly labelled **placeholder gradient** is used so the rest of the
pipeline still runs. Placeholders are *not* valid project results.
""")

code(r"""
PROMPTS_CSV = OUTPUT_DIR / "prompts.csv"
GEN_KEYS = ["prompt", "seed", "model", "steps", "size"]

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

previous = pd.read_csv(PROMPTS_CSV) if PROMPTS_CSV.exists() else None
todo = [r for _, r in prompts.iterrows() if needs_generation(r, previous)]
print(f"{len(prompts) - len(todo)} cached image(s), {len(todo)} to generate")

prompts["generated_by"] = "cache"
if previous is not None and "generated_by" in previous:
    prompts["generated_by"] = prompts.section.map(previous.set_index("section").generated_by).fillna("cache")

if todo:
    try:
        from diffusers import AutoPipelineForText2Image
        from diffusers.utils import logging as diffusers_logging
        from transformers.utils import logging as transformers_logging
        diffusers_logging.set_verbosity_error(); transformers_logging.set_verbosity_error()
        dtype = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(
            MODEL_ID, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None).to(DEVICE)
        pipe.set_progress_bar_config(disable=True)
        for r in todo:
            generator = torch.Generator("cpu").manual_seed(int(r.seed))
            image = pipe(r.prompt, num_inference_steps=int(r.steps), guidance_scale=0.0,
                         width=IMAGE_SIZE, height=IMAGE_SIZE, generator=generator).images[0]
            image.save(r.image_path)
            prompts.loc[prompts.section == r.section, "generated_by"] = MODEL_ID
            print(f"  generated section {r.section}")
        del pipe
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
    ax.set_title(f"Section {r.section}\nenergy {r.rms_level} · bass {r.bass_level} · high {r.high_level}", fontsize=10)
plt.tight_layout(); plt.savefig(FIG_DIR / "section_images.png"); plt.show()
""")

# =============================================================================
# 12. Mappings
# =============================================================================
md(r"""
## 12. Define audio-to-visual mappings

The renderer draws seven layers. Each visual parameter is a small, inspectable formula of the normalized controls (coefficients in `VIS`):

| Layer | Audio control | Visual parameter | Formula |
|---|---|---|---|
| 1. Section image | `section_id` | which image + crossfade | blend over `CROSSFADE_SECONDS` around each boundary |
| 2. Low-frequency geometry | bass, beat pulse | image zoom, ring radius & thickness | `zoom = 1.02 + 0.10·bass + 0.03·beat`, `ring = 0.22 + 0.18·beat`, `thickness = 2 + 12·bass` |
| 3. Mid-frequency geometry | mid | polygon rotation, ribbon amplitude, background rotation | `poly speed = 15 + 90·mid` °/s, `ribbon = 0.02 + 0.10·mid` |
| 4. High-frequency detail | high | number of particles, sparkle jitter | `particles = 40 + 360·high` |
| 5. Overall intensity | RMS | brightness, saturation | `brightness = 0.75 + 0.40·rms` |
| 6. Spectral brightness | centroid | sharpness (unsharp mask) | `sharpness = 0.90 + 0.30·centroid` |
| 7. Events | onset pulse | white flash | `flash = 0.30·onset_pulse` |

Large / slow structures follow the bass; small / fast details follow the highs (the wavelength-inspired grammar from section 10).

Both the audio-conditioned version and the baseline produce a **parameter table with identical columns**; one renderer draws both.
Rotation angles are integrated over time (`cumsum(speed) / FPS`) so speed changes never cause jumps.
""")

code(r"""
def crossfade_plan(times, edges, crossfade=CROSSFADE_SECONDS):
    # For each frame: index of image A, index of image B and blend weight (0 = all A, 1 = all B).
    sec = np.clip(np.searchsorted(edges, times, side="right") - 1, 0, len(edges) - 2)
    img_a, img_b, blend = sec.copy(), sec.copy(), np.zeros_like(times)
    for k, b in enumerate(edges[1:-1]):            # boundary k separates image k and k+1
        in_fade = np.abs(times - b) < crossfade / 2
        img_a[in_fade], img_b[in_fade] = k, k + 1
        blend[in_fade] = (times[in_fade] - (b - crossfade / 2)) / crossfade
    return img_a, img_b, blend


def visual_params(ctrl, edges):
    # ctrl: DataFrame with time_sec, bass, mid, high, rms, centroid, beat_pulse, onset_pulse (all 0-1).
    v, t = VIS, ctrl.time_sec.values
    p = pd.DataFrame({"time_sec": t})
    p["img_a"], p["img_b"], p["blend"] = crossfade_plan(t, edges)
    p["zoom"] = v["zoom_base"] + v["zoom_bass"] * ctrl.bass.values + v["zoom_beat"] * ctrl.beat_pulse.values
    p["bg_angle"] = np.cumsum(v["bg_rot_base"] + v["bg_rot_mid"] * ctrl.mid.values) / FPS
    p["ring_radius"] = v["ring_base"] + v["ring_beat"] * ctrl.beat_pulse.values
    p["ring_thickness"] = v["ring_thick_base"] + v["ring_thick_bass"] * ctrl.bass.values
    p["ring_alpha"] = 0.35 + 0.65 * np.maximum(ctrl.bass.values, ctrl.beat_pulse.values)
    p["poly_angle"] = np.cumsum(v["poly_rot_base"] + v["poly_rot_mid"] * ctrl.mid.values) / FPS
    p["ribbon_amp"] = v["ribbon_amp_base"] + v["ribbon_amp_mid"] * ctrl.mid.values
    p["n_particles"] = (v["particles_min"] + v["particles_range"] * ctrl.high.values).astype(int)
    p["jitter"] = ctrl.high.values
    p["brightness"] = v["brightness_base"] + v["brightness_rms"] * ctrl.rms.values
    p["saturation"] = v["saturation_base"] + v["saturation_rms"] * ctrl.rms.values
    p["sharpness"] = v["sharpness_base"] + v["sharpness_centroid"] * ctrl.centroid.values
    p["flash"] = v["flash_onset"] * ctrl.onset_pulse.values
    return p

audio_params = visual_params(timeline, section_edges)
audio_params.describe().T[["mean", "min", "max"]].round(3)
""")

# =============================================================================
# 13. Renderer
# =============================================================================
md(r"""
## 13. Render the audio-conditioned visualizer

For every frame the renderer reads **one row of the parameter table** (looked up by frame timestamp) and:

1. blends the two current section images (crossfade),
2. zooms and slowly rotates the blend (`cv2.warpAffine`),
3. adjusts sharpness (unsharp mask), saturation and brightness,
4. draws the procedural overlay on a black canvas: bass rings, mid polygon + ribbon, high-frequency particles,
5. adds a soft glow and composites the overlay with a *screen* blend, then adds the onset flash.

Particles have fixed random starting positions (seeded) and drift outward over time, so the particle field is identical in every
version; only *how many* are visible depends on the controls. Frames are streamed straight to an H.264 file, and a small grayscale copy of
each frame is kept to measure **visual motion energy** in section 16.
""")

code(r"""
CENTER = (OUTPUT_SIZE / 2, OUTPUT_SIZE / 2)
MAX_PARTICLES = VIS["particles_min"] + VIS["particles_range"]
_rng = np.random.default_rng(RANDOM_SEED)
PARTICLES = {                                     # fixed particle field shared by all renders
    "angle": _rng.uniform(0, 2 * np.pi, MAX_PARTICLES),
    "r0": _rng.uniform(0.05, 0.7, MAX_PARTICLES),
    "speed": _rng.uniform(0.02, 0.10, MAX_PARTICLES),     # fraction of frame per second
    "size": _rng.choice([1, 1, 2], MAX_PARTICLES),
}

def accent_color(img):
    # Bright overlay colour derived from the image's average colour, so overlays match the artwork.
    c = img.reshape(-1, 3).mean(axis=0)
    c = c / (c.max() + 1e-6)
    return 0.55 * c + 0.45                        # mix with white


def render_frame(p, frame_idx, images):
    S = OUTPUT_SIZE
    # 1. Section image crossfade
    a, b = images[int(p.img_a)], images[int(p.img_b)]
    img = (1 - p.blend) * a + p.blend * b

    # 2. Zoom + slow rotation (bass -> zoom, mid -> rotation speed)
    M = cv2.getRotationMatrix2D(CENTER, p.bg_angle, p.zoom)
    img = cv2.warpAffine(img, M, (S, S), borderMode=cv2.BORDER_REFLECT)

    # 3. Sharpness (centroid), saturation and brightness (RMS)
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = img + (p.sharpness - 1.0) * 2.0 * (img - blur)
    gray = img.mean(axis=2, keepdims=True)
    img = (gray + p.saturation * (img - gray)) * p.brightness

    # 4. Procedural overlay
    ov = np.zeros((S, S, 3), np.float32)
    col = tuple(float(x) for x in accent_color(a if p.blend < 0.5 else b))
    # 4a. Bass rings: large, thick, expand on beats
    for k, scale in enumerate((1.0, 1.45, 1.9)):
        radius = int(p.ring_radius * scale * S / 2 * 1.4)
        thick = max(1, int(p.ring_thickness / (k + 1)))
        cv2.circle(ov, (S // 2, S // 2), radius, tuple(c * p.ring_alpha / (k + 1) for c in col), thick, cv2.LINE_AA)
    # 4b. Mid polygon (rotating hexagon) and ribbon (sine wave)
    ang = np.deg2rad(p.poly_angle) + np.arange(6) * np.pi / 3
    pts = np.stack([S / 2 + 0.28 * S * np.cos(ang), S / 2 + 0.28 * S * np.sin(ang)], 1).astype(np.int32)
    cv2.polylines(ov, [pts], True, tuple(0.6 * c for c in col), 2, cv2.LINE_AA)
    xs = np.linspace(0, S, 120)
    ys = S / 2 + p.ribbon_amp * S * np.sin(xs / S * 4 * np.pi + p.time_sec * 2.0)
    cv2.polylines(ov, [np.stack([xs, ys], 1).astype(np.int32)], False, tuple(0.5 * c for c in col), 2, cv2.LINE_AA)
    # 4c. High-frequency particles: small, many, drifting outward with jitter
    n = int(p.n_particles)
    r = (PARTICLES["r0"][:n] + PARTICLES["speed"][:n] * p.time_sec) % 0.75 * S
    jit = np.random.default_rng(frame_idx).normal(0, 1.5 * p.jitter, (n, 2))
    px = (S / 2 + r * np.cos(PARTICLES["angle"][:n]) + jit[:, 0]).astype(int)
    py = (S / 2 + r * np.sin(PARTICLES["angle"][:n]) + jit[:, 1]).astype(int)
    for x0, y0, sz in zip(px, py, PARTICLES["size"][:n]):
        cv2.circle(ov, (int(x0), int(y0)), int(sz), (1.0, 1.0, 1.0), -1, cv2.LINE_AA)

    # 5. Glow + screen blend + onset flash
    ov = np.clip(ov + 0.8 * cv2.GaussianBlur(ov, (0, 0), 6.0), 0, 1) * VIS["overlay_strength"]
    img = np.clip(img, 0, 1)
    img = 1 - (1 - img) * (1 - ov)
    img = img + p.flash
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
| bass, mid, high, RMS, centroid | measured, normalized | constant `BASELINE_LEVEL` = 0.5 (mid-range of every effect) |
| beat pulse | decays from each detected beat | same decay shape, repeating every `BASELINE_PULSE_PERIOD` = 0.70 s (not the song's tempo) |
| onset flash | onset events × strength | fixed flash every `BASELINE_FLASH_PERIOD` = 2.8 s at half strength |
| brightness | RMS | slow sine cycle (period 10 s) around the mid level |
| section transitions | detected boundaries | equal time windows |

The baseline is meant to look polished, not deliberately bad: it simply removes the audio-conditioning mechanism.
""")

code(r"""
def baseline_controls(times):
    # Preset control signals with the same columns as the audio timeline, built ONLY from time.
    c = pd.DataFrame({"time_sec": times})
    for name in ["bass", "mid", "high", "centroid"]:
        c[name] = BASELINE_LEVEL
    c["rms"] = BASELINE_LEVEL + 0.3 * np.sin(2 * np.pi * times / BASELINE_BRIGHT_PERIOD)
    c["beat_pulse"] = np.exp(-PULSE_DECAY * (times % BASELINE_PULSE_PERIOD))
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
    cols = ["rms", "bass", "mid", "high", "centroid", "beat_pulse", "onset_pulse"]
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
* **Correlation** (Pearson and Spearman) of motion energy with RMS and with onset strength, for both videos.
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
    for feature in ["rms", "onset"]:
        x = timeline[feature].values
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
print(f"Images: {prompts.generated_by.unique().tolist()} | seed {RANDOM_SEED} | {GEN_STEPS} steps\n")
print("Motion energy vs audio (rendered frames, crossfades excluded):")
for f in ["rms", "onset"]:
    print(f"  {f:>5}: audio-conditioned r = {a.loc[f, 'pearson_r']:+.2f} | baseline r = {b.loc[f, 'pearson_r']:+.2f}")
print(f"  best onset->motion lag: audio-conditioned {a.loc['onset', 'best_lag_ms']:+.0f} ms | "
      f"baseline {b.loc['onset', 'best_lag_ms']:+.0f} ms")
print(f"\nOutputs: {FINAL_VIDEO}, {BASELINE_VIDEO}")
""")

md(r"""
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
* **Prompt following is limited.** SD-Turbo is a small, few-step model; images reflect the descriptors only loosely, and the 77-token
  CLIP limit constrains prompt detail.
* **Technical metrics test the implementation, not perception.** Motion energy is one crude output measure.
* **Small survey samples** do not generalize beyond the participants.
* The development clip is a **synthesized** track with very clear sections; real songs will be less tidy.

## 20. Future work

* Run the identical notebook on one or two **real songs** (set `AUDIO_PATH`) to check that the pipeline is not clip-specific.
* Use **image-to-image** generation from Section 1's image to make later section images even more coherent.
* Try a stronger model (SDXL-Turbo) if GPU memory allows, and compare prompt adherence.
* Add the optional **time-shifted** comparison as a second baseline and an accessibility-focused evaluation.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, "genai_audio_visualizer.ipynb")
print(f"Wrote genai_audio_visualizer.ipynb with {len(cells)} cells")
