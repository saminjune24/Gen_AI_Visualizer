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
CLAP_MODEL    = "laion/clap-htsat-unfused"
VIBE_OVERRIDE = None          # e.g. "calm" to force a vibe; None = detect it from the audio

# ---- Sections -------------------------------------------------------------------
N_SECTIONS      = 4           # target number of analysis sections (3-4)
MIN_SECTION_SEC = 4.5         # minimum section length; also the minimum spacing between boundaries
ACTIVITY_WEIGHTS = {"rms": 0.50, "onset": 0.30, "bass": 0.20}
ACTIVITY_SMOOTH_SEC = 1.0     # extra smoothing before looking for change points

# ---- Image generation ---------------------------------------------------------------
MODEL_ID        = "stabilityai/sd-turbo"
IMAGE_SIZE      = 512         # SD-Turbo's native resolution
GEN_STEPS       = 2           # SD-Turbo is designed for 1-4 steps
RANDOM_SEED     = 42          # section N uses seed RANDOM_SEED + N (shared prompt structure keeps them related)
FORCE_REGENERATE = False      # True = ignore cached images and generate again

# ---- Video ----------------------------------------------------------------------------
FPS               = 24
OUTPUT_SIZE       = 512       # square video, pixels
CROSSFADE_SECONDS = 1.0       # length of the blend between section images
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
## 9.1 Detect the vibe (CLAP)

**[CLAP](https://huggingface.co/laion/clap-htsat-unfused)** (Contrastive Language-Audio Pretraining) is a pretrained model that embeds audio
and text in the same space, so it can score how well the clip matches each text label without any training (**zero-shot**).
The clip is cut into 10-second windows (CLAP's training length), each window is scored against five vibe descriptions, and the
probabilities are averaged. The winning vibe sets the **theme and palette** shared by every section prompt.

This is a genre-level guess, not a validated mood classifier. If CLAP cannot be loaded, a simple rule based on tempo, onset rate and
brightness is used instead. Set `VIBE_OVERRIDE` in the config to force a vibe.
""")

code(r"""
VIBES = {   # vibe -> text description CLAP compares the audio against
    "party":  "energetic party dance music",
    "calm":   "calm relaxing ambient music",
    "dark":   "dark intense aggressive music",
    "dreamy": "dreamy emotional melancholic music",
    "epic":   "epic cinematic orchestral music",
}

def vibe_scores_clap(y):
    from transformers import ClapModel, ClapProcessor
    model = ClapModel.from_pretrained(CLAP_MODEL).eval()
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    clap_sr = 48000
    y48 = librosa.resample(y, orig_sr=SR, target_sr=clap_sr)
    win = 10 * clap_sr
    windows = [y48[i:i + win] for i in range(0, max(1, len(y48) - win + 1), win)]
    inputs = processor(text=list(VIBES.values()), audio=windows, sampling_rate=clap_sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        probs = model(**inputs).logits_per_audio.softmax(dim=-1).mean(dim=0).numpy()   # average over windows
    return dict(zip(VIBES, probs))

def vibe_scores_rules():
    # Fallback: fast + busy -> party, sparse + dark -> calm, otherwise dreamy.
    onset_rate = len(onset_times) / DURATION
    if tempo >= 110 and onset_rate >= 3:
        pick = "party"
    elif onset_rate < 1.5 and centroid.mean() < 1500:
        pick = "calm"
    else:
        pick = "dreamy"
    return {v: float(v == pick) for v in VIBES}

try:
    vibe_scores = vibe_scores_clap(y)
    VIBE_METHOD = f"CLAP zero-shot ({CLAP_MODEL})"
except Exception as e:
    print(f"CLAP unavailable ({type(e).__name__}: {e}); using rule-based vibe")
    vibe_scores, VIBE_METHOD = vibe_scores_rules(), "rule-based fallback"

VIBE = VIBE_OVERRIDE or max(vibe_scores, key=vibe_scores.get)
print(f"Vibe: {VIBE}  ({VIBE_METHOD}{', overridden' if VIBE_OVERRIDE else ''})")

fig, ax = plt.subplots(figsize=(7, 2.6))
names = list(vibe_scores)
ax.barh(names, [vibe_scores[n] for n in names],
        color=[COLORS["rms"] if n == VIBE else "#c3c2b7" for n in names], height=0.6)
for k, n in enumerate(names):
    ax.text(vibe_scores[n] + 0.01, k, f"{vibe_scores[n]:.0%}", va="center", fontsize=9)
ax.set(xlim=(0, 1.1), xlabel="probability", title="Vibe scores (CLAP, averaged over 10 s windows)")
ax.invert_yaxis(); ax.grid(axis="y", visible=False)
plt.tight_layout(); plt.savefig(FIG_DIR / "vibe.png"); plt.show()
""")

# =============================================================================
# 10. Prompts
# =============================================================================
md(r"""
## 10. Build GenAI prompts from section features

The diffusion model never sees the waveform. Instead each section goes through a transparent chain:

`vibe + section statistics → low / medium / high levels → visual descriptors → text prompt → image`

Every prompt has three parts:

1. **Fixed structure** (same for all sections): *a glowing star inside a larger star with radiating light lines, centred on a dark
   background*. This is deliberate: the renderer (section 13) animates the image **part by part by radius** (core, inner star,
   outer star), so the images need a centred, layered radial structure.
2. **Vibe theme** (same for all sections, from section 9.1): colours and lighting, e.g. *neon club lights* for party.
3. **Section descriptors** (different per section): one short phrase per measured feature.

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

# 1. Fixed structure shared by every section (the renderer animates these parts)
STRUCTURE = ("luminous digital art, centered radial mandala, glowing star inside a larger star, "
             "radiating light lines, dark background")

# 2. Vibe theme shared by every section
VIBE_THEMES = {
    "party":  "neon club lights, laser beams, magenta and cyan glow",
    "calm":   "soft mist, pastel light, gentle glow",
    "dark":   "deep red and black, smoke, harsh light",
    "dreamy": "nebula haze, violet and pink glow, soft bokeh",
    "epic":   "golden light rays, vast cosmic scale",
}

# 3. One short descriptor per feature and level
DESCRIPTORS = {
    "rms":      {"low": "sparse dim calm", "medium": "layered moderately bright", "high": "dense intense glow"},
    "bass":     {"low": "small central core", "medium": "round central core", "high": "huge glowing core, thick rings"},
    "mid":      {"low": "simple star shape", "medium": "layered star points", "high": "many layered star points, ribbons"},
    "high":     {"low": "smooth surfaces", "medium": "some fine lines", "high": "fine particles, thin sparkling lines"},
    "centroid": {"low": "soft rounded edges", "medium": "balanced edges", "high": "crisp sharp edges"},
    "onset_density_rel": {"low": "few elements", "medium": "several accents", "high": "fragmented bursts"},
}
FEATURES_FOR_PROMPT = ["rms", "bass", "mid", "high", "centroid", "onset_density_rel"]

def build_prompt(row):
    levels = {f: level(row[f]) for f in FEATURES_FOR_PROMPT}
    phrases = [DESCRIPTORS[f][levels[f]] for f in FEATURES_FOR_PROMPT]
    prompt = ", ".join([STRUCTURE, VIBE_THEMES[VIBE]] + phrases)
    description = (f"A {VIBE} song. This section has {levels['rms']} energy with {levels['bass']} low-frequency presence, "
                   f"{levels['mid']} mid-frequency activity, {levels['high']} high-frequency detail, "
                   f"{levels['centroid']} spectral brightness and {levels['onset_density_rel']} onset density.")
    return prompt, description, levels

records = []
for _, row in sections.iterrows():
    prompt, description, levels = build_prompt(row)
    s = int(row.section)
    records.append({"section": s, "vibe": VIBE, **{f"{k}_level": v for k, v in levels.items()},
                    "section_description": description, "prompt": prompt,
                    "seed": RANDOM_SEED + s, "model": MODEL_ID, "steps": GEN_STEPS, "guidance_scale": 0.0,
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

prompts[["section", "vibe", "rms_level", "bass_level", "mid_level", "high_level", "centroid_level", "onset_density_rel_level"]]
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

**Consistency vs. variety:** every prompt shares the same structure and vibe theme, which keeps the images related. Each section uses
its own seed (`RANDOM_SEED + section`), so sections with similar statistics still get visibly different artwork.

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
| Section image | section id | crossfade | over `CROSSFADE_SECONDS` around each boundary |

Rotation angles are integrated over time (`cumsum(speed) / FPS`) so speed changes never cause jumps. Both the audio-conditioned version
and the baseline produce a **parameter table with identical columns**, and one renderer draws both.
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
    # ctrl: DataFrame of controls (all 0-1 except drum_hits). Returns one row of visual parameters per frame.
    v, t = VIS, ctrl.time_sec.values
    deg_per_frame = lambda speed: np.cumsum(speed) / FPS           # integrate deg/s -> degrees
    p = pd.DataFrame({"time_sec": t})
    p["img_a"], p["img_b"], p["blend"] = crossfade_plan(t, edges)
    p["zoom"] = v["zoom_base"] + v["zoom_bass"] * ctrl.bass.values
    p["core_scale"] = 1 + v["core_scale_bass"] * ctrl.bass_stem.values + v["core_scale_beat"] * ctrl.beat_pulse.values
    p["core_angle"] = deg_per_frame(np.full(len(t), v["core_rot_base"]))
    p["inner_angle"] = deg_per_frame(v["inner_rot_base"] + v["inner_rot_lead"] * ctrl.lead.values)
    p["inner_twist"] = v["swirl_lead"] * ctrl.lead.values
    p["outer_angle"] = -deg_per_frame(v["outer_rot_base"] + v["outer_rot_other"] * ctrl.other.values)
    p["outer_scale"] = 1 + v["outer_scale_beat"] * ctrl.beat_pulse.values
    p["ripple_amp"] = v["ripple_high"] * ctrl.high.values
    p["ripple_phase"] = 2 * np.pi * v["ripple_speed"] * t
    p["lines_gain"] = v["lines_base"] + v["lines_drums"] * ctrl.drum_pulse.values
    p["lines_stretch"] = 1 + v["lines_stretch"] * ctrl.drum_pulse.values
    p["lights_gain"] = v["lights_base"] + v["lights_drums"] * ctrl.drum_pulse.values + v["lights_rms"] * ctrl.rms.values
    p["light_hue"] = (v["light_hue_step"] * ctrl.drum_hits.values) % 1.0
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

1. **Builds one warp map in polar coordinates.** Every output pixel at radius *r* and angle *θ* samples the source image at a rotated,
   scaled position. The rotation and scale are a soft mix of the core / inner / outer settings for that radius, plus the twist (extra
   rotation growing across the inner star) and the high-frequency ripple. One `cv2.remap` call then moves all parts at once.
2. Crossfades the two current section images and warps them with that map. The **line-art** and **light-spot** layers are warped with
   the same map (so they stay attached to the artwork), with an extra outward stretch on drum hits.
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

def warp_maps(p, extra_scale=1.0):
    # Source sampling coordinates for every output pixel (polar warp: per-part rotation, scale, twist, ripple).
    w_core, w_inner, w_outer = part_weights(RADIUS, VIS["core_radius"] * p.core_scale, VIS["inner_radius"])
    twist = p.inner_twist * w_inner * (RADIUS - VIS["core_radius"])
    angle = np.deg2rad(w_core * p.core_angle + w_inner * p.inner_angle + w_outer * p.outer_angle) + twist
    scale = (w_core * p.core_scale + w_inner + w_outer * p.outer_scale) * p.zoom * extra_scale
    r_src = RADIUS / scale + p.ripple_amp * np.sin(VIS["ripple_waves"] * 2 * np.pi * RADIUS - p.ripple_phase)
    theta_src = THETA - angle
    map_x = (CX + r_src * np.cos(theta_src) * (S / 2)).astype(np.float32)
    map_y = (CY + r_src * np.sin(theta_src) * (S / 2)).astype(np.float32)
    return map_x, map_y

def render_frame(p, frame_idx, images):
    a, b, t = int(p.img_a), int(p.img_b), p.blend
    mix = lambda key: (1 - t) * layers[a][key] + t * layers[b][key]

    # 1-2. Crossfade, then move every part of the image with one polar warp map
    mx, my = warp_maps(p)
    img = cv2.remap((1 - t) * images[a] + t * images[b], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sx, sy = warp_maps(p, extra_scale=p.lines_stretch)            # lines + lights stretch outward on drum hits
    lines = cv2.remap(mix("lines"), sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    lights = cv2.remap(mix("lights"), sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 3. Sharpness (centroid), saturation and brightness (RMS)
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = img + (p.sharpness - 1.0) * 2.0 * (img - blur)
    gray = img.mean(axis=2, keepdims=True)
    img = np.clip((gray + p.saturation * (img - gray)) * p.brightness, 0, 1)

    # 4. Light spots take on a colour that changes on every drum hit (they glow brighter on hits)
    g = np.clip(lights * p.lights_gain, 0, 1)[..., None]
    img = img * (1 - g) + g * hue_to_rgb(p.light_hue)
    # 5. Glowing line art (screen blend), particles, onset flash
    line_col = accent_color(images[a] if t < 0.5 else images[b])
    glow = lines + 0.8 * cv2.GaussianBlur(lines, (0, 0), 4.0)
    overlay = np.clip(glow[..., None] * line_col * p.lines_gain, 0, 1).astype(np.float32)
    n = int(p.n_particles)
    r = (PARTICLES["r0"][:n] + PARTICLES["speed"][:n] * p.time_sec) % 0.75 * S
    jit = np.random.default_rng(frame_idx).normal(0, 1.5 * p.jitter, (n, 2))
    px = (S / 2 + r * np.cos(PARTICLES["angle"][:n]) + jit[:, 0]).astype(int)
    py = (S / 2 + r * np.sin(PARTICLES["angle"][:n]) + jit[:, 1]).astype(int)
    for x0, y0, sz in zip(px, py, PARTICLES["size"][:n]):
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
| section transitions | detected boundaries | equal time windows |

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
            "drums", "bass_stem", "other", "vocals", "lead", "drum_pulse", "drum_hits"]
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
print(f"Vibe: {VIBE} ({VIBE_METHOD})")
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
* **The vibe is a coarse zero-shot guess** from five hand-written labels, not a validated mood classifier.
* **Parts are found by radius**, which works because the prompts ask for a centred radial composition; arbitrary scenes
  (e.g. a crowd) would need a segmentation model, and people are not animated.
* **Prompt following is limited.** SD-Turbo is a small, few-step model; images reflect the descriptors only loosely, and the 77-token
  CLIP limit constrains prompt detail.
* **Technical metrics test the implementation, not perception.** Motion energy is one crude output measure.
* **Small survey samples** do not generalize beyond the participants.
* The development clip is a **synthesized** track with very clear sections; real songs will be less tidy.

## 20. Future work

* Run the identical notebook on one or two **real songs** (set `AUDIO_PATH`) to check that the pipeline is not clip-specific.
* Use **image-to-image** generation from Section 1's image to make later section images even more coherent.
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
