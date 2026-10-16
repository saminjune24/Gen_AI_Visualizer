# Section-Aware Procedural GenAI Audio Visualizer

A music visualizer built in a single Jupyter notebook. It listens to a 30-second clip, works out what is happening in it (beats,
instruments, key, mood, where the song changes), generates a few abstract images that match the song, and animates them frame by
frame so they move with the music. The output is an MP4 with the original audio.

Final project for MSDS 466 (Applied Generative AI for Enterprises), by Samin Bahizad and Wendy Zhang.

<p align="center">
  <img src="outputs/demo_track/figures/section_images.png" width="900" alt="The four generated section images for the demo track">
</p>

**Demo:** [`visualizer.mp4`](outputs/demo_track/videos/visualizer.mp4) (audio-driven) vs. [`baseline.mp4`](outputs/demo_track/videos/baseline.mp4)
(not audio-driven). GitHub does not play video inline, so click **View raw** or download the file to watch it with sound.

## Research question

> Can audio-derived features guide generative artwork and animation so that the result feels more synchronized and connected to the
> music than a visualizer that ignores the audio?

**Hypothesis:** the motion in the audio-driven visualizer will line up with the music more closely than the baseline's.

The **baseline** uses the same images and effects, but runs them on fixed timers instead of the music. We compare the two with
measurements on the rendered frames (see [Evaluation](#evaluation)).

## Repository layout

```
Gen_AI_Visualizer/
├── genai_audio_visualizer.ipynb  ← the whole project, run top to bottom
├── requirements.txt
├── sample_inputs/                ← input clips (see sample_inputs/README.md)
└── outputs/<song>/               ← one folder per song
    ├── videos/                   ← visualizer.mp4 and baseline.mp4
    ├── images/                   ← generated section images
    ├── figures/                  ← plots from the notebook
    └── tables/                   ← features, prompts, sections, metrics, test report
```

There are no separate scripts. The two real songs are copyrighted, so their audio and videos are not in the repo. Each run also
creates a git-ignored `cache/` folder with the clip audio and separated instruments.

## Quick start

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter nbconvert --to notebook --execute --inplace genai_audio_visualizer.ipynb
```

You can also open the notebook in Jupyter Lab or VS Code with the `.venv` kernel and run all cells. No system FFmpeg is needed.
It uses an NVIDIA GPU, an Apple-silicon GPU or the CPU, whichever is available.

The first run downloads about 4.5 GB of models. After that, a full run takes about 3 minutes on an M-series MacBook. Generated images
are cached and only regenerated when the prompts change.

**Using another song:** put the file in `sample_inputs/`, add it to `SAMPLE_SONGS` in the configuration cell (section 2) and set `SONG`:

```python
SAMPLE_SONGS = {   # name: (file, clip start in seconds)
    "demo_track": ("sample_inputs/demo_track.wav", 0),
    "my_song":    ("sample_inputs/my_song.mp3", 45),
}
SONG = "my_song"
```

Results go to `outputs/my_song/`. Pick a 30-second window with some contrast (a build-up, drop or breakdown), since that is what the
section detection reacts to.

## How it works

| Part | Notebook sections | What happens |
|---|---|---|
| **A. Audio analysis** | 3–8 | Extract beats, energy, brightness and bass/mid/high levels. Split the clip into drums, bass, vocals and other instruments (Demucs), and label each drum hit as kick, snare or hi-hat. Resample everything to one row per video frame (24 fps). |
| **B. Song understanding** | 9–14 | Find 3–4 sections where the music changes. Classify the song into one of eight archetypes (party, euphoric, chill, melancholic, dark_synth, aggressive, groove, cinematic). Flag big beat changes between sections. Give each section a motion profile. |
| **C. Images** | 15–17 | Build one prompt per section and generate 3–4 images with SD-Turbo. Split each image into parts (core, inner star, outer ring, line art, light spots) that can move separately. |
| **D. Animation** | 18–21 | Animate the images from the per-frame timeline and add the original audio back. |
| **E. Evaluation** | 22–25 | Render the baseline, measure audio–visual alignment, and test the analysis against the demo track's known answers. |
| **F. Wrap-up** | 26–28 | Results summary, test report and conclusion. |

All settings are in the configuration cell (section 2). Every `check(...)` line in the notebook is a test that prints `[PASS]`,
`[WARN]` or `[FAIL]`. Pipeline tests stop the run when something is broken; quality tests only warn. Section 27 collects them in
`tables/test_report.csv`.

### Song understanding

The archetype comes from two sources averaged 50/50:

* **Acoustic features** matched against a hand-written prototype for each archetype: key (major/minor), how driving the drums are,
  how tight they sit on the grid, brightness, which instrument dominates, energy and distortion.
* **CLAP**, a pretrained audio–text model, scoring the clip against a one-line description of each archetype.

Whisper transcribes the vocals and a sentiment model rates them. An upbeat song with mostly negative lyrics is marked **bittersweet**,
which shifts the colours towards cold blue.

Each section blends the motion profiles of its top two archetypes, so a quiet breakdown in an upbeat song can calm down.

### Motion profiles

| Profile | What it looks like |
|---|---|
| party | hard beat zooms, many mirror segments, fast colour changes, sharp white flashes |
| euphoric | bouncy, bright and saturated, moderate flashes |
| chill | slow drift, gentle camera sway, few segments, no flashes |
| melancholic | minimal: slow rotation, muted colours, long transitions |
| dark_synth | moody and steady: high contrast, slow colour drift, deep swirl |
| aggressive | chaotic: the image shatters into tiles and spins on drum hits |
| groove | bouncy: the centre swells with the bass, thick lines, swinging camera |
| cinematic | deep: fast zoom, big swirling transitions, slow brightness swells |

### Images

Each prompt has four parts, each chosen by a rule:

| Part | Chosen by | Example |
|---|---|---|
| Art style | archetype | party → *retro-futuristic cyberpunk neon vector art* |
| Subject | section energy and rhythm | high and driving → *intricate mandala of fractured crystal prisms* |
| Colours | key and brightness | minor and dark → *deep indigo, midnight obsidian, teal accents* |
| Lighting | brightness and rhythm | dark → *volumetric fog, soft bioluminescence* |

The first section's image is generated from text. Later sections start from the previous image so the artwork evolves, unless there was
a big beat change, in which case they start fresh. All prompts and seeds are saved to `tables/prompts.csv`.

<p align="center">
  <img src="outputs/demo_track/figures/layers.png" width="650" alt="Each section image split into parts, line art and light spots">
</p>

### Animation

Every frame is drawn from the timeline with simple formulas. In the default style:

| Music | What you see |
|---|---|
| Loudness | endless zoom speed, brightness, colour cycling speed |
| Kick | zoom punch and the centre swells |
| Snare | kaleidoscope twist, white flash, light spots flare |
| Hi-hats | sparkling particles |
| Vocals | the inner star glows magenta |
| Bass / vocals / other | core / inner star / outer ring glow orange / magenta / cyan |
| Instrument enters or leaves | a ring blooms and the number of mirror segments changes |
| New section | the next image melts in with a swirl, no hard cuts |

Set `VISUAL_STYLE = "modes"` for an alternative style where each shape family has its own motion.

### Evaluation

* **Baseline:** same images and effects, but all levels are constant, pulses fire every 0.7 s (deliberately not the song's tempo),
  and sections change at equal intervals.
* **Alignment metrics** are measured on the rendered frames, not on our own parameters. Motion is the average change between consecutive
  frames. We correlate it with the audio, and we measure **event-locked contrast**: motion right at each kick, snare or vocal accent
  divided by motion just before it. Above 1 means the picture visibly reacts, and the baseline should stay near 1.
* **Ground-truth test:** we synthesized the demo track, so we know every beat, hit, section and instrument change, and we score the
  analysis against them.

## Results

Three 30-second clips: the demo track and two real songs (see [`sample_inputs/README.md`](sample_inputs/README.md)).

| | Demo track | Naïka – *BARELY, BARELY* | Twenty One Pilots – *Drag Path* |
|---|---|---|---|
| Tempo / key | 117.5 BPM, A minor | 136 BPM, D# major | 96 BPM, E major |
| **Archetype** (runner-up) | **party** (groove) | **groove** (euphoric), bittersweet lyrics | **euphoric** (cinematic) |
| Section profiles | dark_synth → groove → party → dark_synth | euphoric → groove → groove → groove | euphoric → chill → chill → euphoric |
| Art style | cyberpunk neon vector art | 70s funk poster | pop-art illustration |
| **Onset ↔ motion r** (audio vs. baseline) | **0.76** vs. 0.00 | **0.36** vs. −0.12 | **0.42** vs. 0.01 |
| Loudness ↔ motion r | 0.33 vs. −0.07 | 0.43 vs. −0.15 | 0.69 vs. 0.15 |
| **Kick contrast** | 1.25× vs. 0.97× | **1.43×** vs. 0.84× | **1.85×** vs. 1.10× |
| Snare contrast | 1.07× vs. 0.99× | 1.20× vs. 0.80× | 1.21× vs. 0.89× |
| Vocal contrast | (no vocals) | 0.96× vs. 0.93× | 1.44× vs. 0.92× |

<p align="center">
  <img src="outputs/demo_track/figures/alignment.png" width="750" alt="Motion energy vs onset strength for both videos, and the cross-correlation">
</p>

**What we learned:**

* On every clip, motion follows the music much more closely in the audio-driven video than in the baseline.
* The effect is strongest on the clean electronic demo. On the real songs, sync improved a lot once kicks, snares, hi-hats and vocals
  each got their own effect instead of everything firing on every hit.
* Each clip got a different archetype, and with it a different style, palette and motion. The *Drag Path* breakdown is correctly
  classified as *chill*.
* The ground-truth test matches tempo, hit timing, sections and instrument changes. The kick/snare/hi-hat **labels** are often wrong,
  which is why the snare reaction is the weakest result. This is the main open issue.
* We treat the onset correlation as the fairer test, because louder sections also get busier images, which inflates the loudness numbers.
* Smoother transitions cost some beat-locking. It's a trade-off between a calmer look and a punchier one.

These numbers show the system works as designed. They don't prove that people *perceive* better synchronization; testing that with viewers is a next step.

## Models

| Model | Used for |
|---|---|
| `stabilityai/sd-turbo` | generating the section images |
| Demucs `htdemucs` | splitting drums, bass, vocals and other |
| `laion/clap-htsat-unfused` | archetype and instrument tags |
| `openai/whisper-small` | transcribing the vocals |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | lyric sentiment |

Plus librosa, NumPy, SciPy, pandas, OpenCV, Matplotlib, Hugging Face `diffusers` / `transformers`, and FFmpeg via `imageio-ffmpeg`.
Check the Stability AI community license before publishing generated images.

## Main settings

All in the configuration cell (section 2):

| Setting | Default | What it does |
|---|---|---|
| `SONG`, `SAMPLE_SONGS` | `"demo_track"` | input file and clip start |
| `VISUAL_STYLE` | `psychedelic` | or `modes` |
| `ARCHETYPE_OVERRIDE` | auto | force an archetype |
| `SECTION_PROFILE_WEIGHT` | 0.4 | how much a section's own sound changes its motion |
| `CLAP_TEMPERATURE` | 2.0 | how much CLAP's confidence is softened |
| `N_SECTIONS`, `MIN_SECTION_SEC` | 4, 4.5 s | number of sections and their minimum length |
| `DRAMATIC_THRESHOLD` | 0.5 | how big a beat change must be to start a new image |
| `IMG2IMG_STRENGTH` | 0.6 | how much each image may differ from the previous one |
| `VIS`, `PROFILES` | — | animation settings and the eight motion profiles (section 14) |
| `FORCE_REGENERATE` | `False` | regenerate images even if cached |

## Limitations

* **Nothing is trained on labelled data.** Archetypes, profiles and thresholds are design choices tuned by eye on three clips.
* **Features are estimates.** Beat tracking can lock onto half or double tempo, key detection can be wrong, and instruments leak
  between separated tracks.
* **Sections are change points**, not verses or choruses.
* **Drum labels are the weak link**, as the ground-truth test shows.
* **SD-Turbo follows prompts loosely**: it gets the style and colours right more often than the subject.
* **Three clips is a tiny test set**, and we have not tested with viewers yet.
* **Scope:** only 3–4 images per song, and all motion comes from warping them. No frame-by-frame diffusion, real-time rendering or 3D.

## Future work

* Run a viewer survey comparing the two versions.
* Test more songs and genres, with some hand-labelled archetypes to check the classifier.
* Train a small archetype classifier on CLAP embeddings once labelled data exists.
* Use segmentation masks (e.g. Segment Anything) to animate any object in an image, not just rings around the centre.

## Modification log

We built the project one milestone at a time, then improved it in a loop: render, measure, change one thing, measure again.

| # | Problem | Change | Result |
|---|---|---|---|
| 1 | motion lagged onsets by 250 ms | keep onset peaks when resampling to 24 fps | lag 0 ms |
| 2 | frames washed out to white | softer edges, fewer light spots | clean frames |
| 3 | images looked alike across songs | song-based seeds, chained images, archetype prompts | three distinct styles |
| 4 | the demo's drop at 24 s was missed | use the largest single change, not the sum | detected |
| 5 | too many instrument changes | smoothing + 2 s minimum | matches the demo's ground truth |
| 6 | sentiment model crashed | switched to RoBERTa | lyrics analysis works |
| 7 | every song came out "euphoric" | soften CLAP + consistent label wording | party, groove, euphoric |
| 8 | transitions and hits looked like cuts | softer attack, longer morphs | smooth, at some cost to sync |
| 9 | colour cycling erased each palette | colours swing around the song's palette | palettes stay recognizable |
| 10 | real songs felt out of sync | separate effects for kick, snare, hi-hat, vocals | Naïka onset r 0.21 → 0.36 |
| 11 | snares were invisible | snare twist + flash | 1.20× vs 0.80× baseline |
| 12 | analysis never checked against known answers | ground-truth test + test report | drum labels found to be the open issue |

## References

**Audio analysis and music information retrieval**

* McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal
  analysis in Python. *Proceedings of the 14th Python in Science Conference.* https://librosa.org/doc/latest/index.html
* Ellis, D. P. W. (2007). Beat tracking by dynamic programming. *Journal of New Music Research, 36*(1), 51–60.
* Bello, J. P., Daudet, L., Abdallah, S., Duxbury, C., Davies, M., & Sandler, M. B. (2005). A tutorial on onset detection in music
  signals. *IEEE Transactions on Speech and Audio Processing, 13*(5), 1035–1047.
* Krumhansl, C. L. (1990). *Cognitive Foundations of Musical Pitch.* Oxford University Press.
* Kim, Y. E., Schmidt, E. M., Migneco, R., Morton, B. G., Richardson, P., Scott, J., Speck, J. A., & Turnbull, D. (2010). Music emotion
  recognition: A state of the art review. *Proceedings of ISMIR 2010.*
* Choi, K., Fazekas, G., Sandler, M., & Cho, K. (2017). Convolutional recurrent neural networks for music classification.
  *Proceedings of ICASSP 2017.*

**Generative and pretrained models**

* Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-resolution image synthesis with latent diffusion models.
  *Proceedings of CVPR 2022.*
* Sauer, A., Lorenz, D., Blattmann, A., & Rombach, R. (2023). Adversarial diffusion distillation. *arXiv:2311.17042.*
  SD-Turbo model card: https://huggingface.co/stabilityai/sd-turbo
* Rouard, S., Massa, F., & Défossez, A. (2023). Hybrid transformers for music source separation. *Proceedings of ICASSP 2023.*
  https://github.com/adefossez/demucs
* Wu, Y., Chen, K., Zhang, T., Hui, Y., Berg-Kirkpatrick, T., & Dubnov, S. (2023). Large-scale contrastive language-audio pretraining
  with feature fusion and keyword-to-caption augmentation. *Proceedings of ICASSP 2023.* https://huggingface.co/laion/clap-htsat-unfused
* Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak
  supervision. *Proceedings of ICML 2023.* https://huggingface.co/openai/whisper-small
* Loureiro, D., Barbieri, F., Neves, L., Espinosa Anke, L., & Camacho-Collados, J. (2022). TimeLMs: Diachronic language models from
  Twitter. *Proceedings of ACL 2022: System Demonstrations.* https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

**Software and documentation**

* Hugging Face Diffusers: https://huggingface.co/docs/diffusers
* Hugging Face Transformers: https://huggingface.co/docs/transformers
* Bradski, G. (2000). The OpenCV library. *Dr. Dobb's Journal of Software Tools.* https://opencv.org
* FFmpeg: https://ffmpeg.org

**Inspiration**

* python-audio-reactive-visualizer (LibROSA + OpenCV): https://github.com/actionoise/python-audio-reactive-visualizer
* Deforum Stable Diffusion: https://github.com/deforum/deforum-stable-diffusion
