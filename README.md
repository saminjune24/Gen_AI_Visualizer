# Section-Aware Procedural GenAI Audio Visualizer

A single Jupyter Notebook ([`genai_audio_visualizer.ipynb`](genai_audio_visualizer.ipynb)) that turns a short music clip into an
audio-reactive video. It combines **generative AI** (SD-Turbo creates one abstract image per song section, prompted from that
section's measured audio features) with **procedural animation** (beats, bass/mid/high energy, RMS, onsets and spectral brightness
and the separated **instrument stems** (drums / bass / vocals / other, via Demucs) move the **parts of each generated image**:
the core grows with the bass, the inner star twists with the vocals, the outer star counter-rotates with the other instruments,
line art and light spots flash and change colour on drum hits). A zero-shot audio model (CLAP) picks the song's **vibe**, which sets
the theme and palette of every prompt. It compares the result against a **non-audio-conditioned
baseline** that uses the same images and effects.

**Research question:** can audio-derived features guide generative artwork and procedural animation to produce a visualization that
feels more synchronized and connected to the music than a non-audio-conditioned visualizer?

## Watch the result

* **[outputs/visualizer_audio.mp4](outputs/visualizer_audio.mp4)**: final audio-synced visualizer (with sound)
* **[outputs/baseline_audio.mp4](outputs/baseline_audio.mp4)**: non-audio-conditioned baseline (same song, same images)

On GitHub, open the file and click **View raw** (or download it) to play it with sound. The current videos use the synthesized
30 s demo track; set `AUDIO_PATH` to render the same pipeline on a real song.

## Pipeline

```
audio -> features -> sections -> prompts -> SD-Turbo images -> frame-level control timeline
      -> animation (audio-conditioned)  -> MP4 + audio
      -> animation (preset baseline)     -> MP4 + audio
      -> technical alignment checks + survey design
```

| Notebook section | What happens |
|---|---|
| 3. Configuration | every tunable parameter in one cell |
| 4. Load audio | file or a synthesized 30 s, 4-part demo track |
| 5–7. Features | onset strength, beats, RMS, spectral centroid, low/mid/high band energy, **Demucs instrument stems + drum hits**, then smoothing, robust normalization and a 24 fps control timeline |
| 8–9. Sections | change-point detection on an activity curve (equal-window fallback) plus a per-section summary |
| 9.1 Vibe | CLAP zero-shot vibe (party / calm / dark / dreamy / epic) → prompt theme |
| 10–11. GenAI | structure + vibe + feature-driven prompts and SD-Turbo generation, with images cached in `outputs/images/` |
| 11.1 Layers | each image split into core / inner star / outer star (radial masks), line art (edges) and light spots |
| 12–14. Render | polar-warp renderer that moves each image part with its own signal, then FFmpeg muxes in the original audio |
| 15. Baseline | same images/effects, preset fixed-rate controls that never read the audio |
| 16. Checks | motion energy from rendered frames, correlations, cross-correlation lag, QA |
| 17. Survey | counterbalanced A/B design, response template and analysis helper |

## Run locally (macOS / Linux)

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab genai_audio_visualizer.ipynb        # or open it in VS Code with the .venv kernel
# headless:
.venv/bin/jupyter nbconvert --to notebook --execute --inplace genai_audio_visualizer.ipynb
```

* Image generation uses CUDA if available, Apple-silicon MPS otherwise, then CPU. The first run downloads SD-Turbo (~2.5 GB),
  the Demucs model (~80 MB, runs on CPU: ~15 s per 30 s clip) and CLAP (~600 MB).
* The section images are committed, so re-running the notebook **does not regenerate them** unless the prompts change or
  `FORCE_REGENERATE = True`.
* A full run takes about 1.5 minutes on an Apple-silicon laptop when images are cached.
* To use your own song, set `AUDIO_PATH` (and optionally `CLIP_START`) in the configuration cell, or pass them as environment
  variables without editing the notebook:

  ```bash
  AUDIO_PATH="$HOME/Downloads/song.mp3" CLIP_START=28 .venv/bin/jupyter nbconvert --to notebook --execute \
      genai_audio_visualizer.ipynb --output-dir outputs/my_song --output notebook_run
  ```

  Each song writes to its own folder, `outputs/<song_name>/` (videos, images, prompts, metrics, figures). Song audio and
  song videos are git-ignored.

## Run on Google Colab

Upload the notebook, choose **Runtime → Change runtime type → T4 GPU**, and Run all. The first code cell installs the missing packages.

## Outputs (`outputs/`)

| File | Committed | Content |
|---|---|---|
| `images/section_N.png` | yes | generated section images |
| `prompts.csv` | yes | prompt, seed, model, steps and size for every image (reproducibility) |
| `sections.csv` | yes | section boundaries and feature summary |
| `metrics.csv` | yes | correlations and lag for both videos |
| `figures/*.png` | yes | diagnostic plots |
| `survey_template.csv` | yes | empty survey response sheet |
| `visualizer_audio.mp4`, `baseline_audio.mp4` | yes | final videos **with the song** |
| `audio_clip.wav`, `preview.mp4` | no | intermediate files |

## Notes on wording

Spectral centroid is a *brightness* proxy, not pitch. RMS is an *energy* proxy, not perceived loudness. Sections are *analysis
sections*, not verses or choruses. The "large forms for bass, fine detail for highs" mapping is a wavelength-*inspired* design choice.
The technical metrics check the implementation; the survey measures perception.

## References

* LibROSA: https://librosa.org/doc/latest/index.html
* SD-Turbo model card: https://huggingface.co/stabilityai/sd-turbo
* Diffusers text-to-image: https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/text2img
* Inspiration: https://github.com/actionoise/python-audio-reactive-visualizer, https://github.com/deforum/deforum-stable-diffusion
