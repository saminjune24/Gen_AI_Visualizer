# Sample inputs

| File | In the repo? | Clip used | Why we used it |
|---|---|---|---|
| `demo_track.wav` | yes | 0:00–0:30 | A 30 s electronic track we synthesized in NumPy (section 4 of the notebook). License-free, and since we built it we know exactly what happens when, so it works as ground truth: pad only (0–8 s), kick + bass enter (8 s), hi-hats / clap / arpeggio join (16 s), kick + bass drop out (24 s). |
| `naika_barely_barely.mp3` | no (copyrighted) | 0:28–0:58 | Naïka – *BARELY, BARELY*. Pop with vocals; a quiet build-up that jumps in energy at ~0:37. |
| `twenty_one_pilots_drag_path.mp3` | no (copyrighted) | 1:45–2:15 | Twenty One Pilots – *Drag Path*. Loud section → breakdown where drums and bass drop out (~1:57) → drums slam back in (~2:06). |

The two songs are only on our own machines. To reproduce those runs, put your own copy of the song in this folder with the same
file name and run, from the repo root:

```bash
.venv/bin/python tools/render_song.py sample_inputs/naika_barely_barely.mp3 --start 28
.venv/bin/python tools/render_song.py sample_inputs/twenty_one_pilots_drag_path.mp3 --start 105
```

**Using your own song:** drop any `.wav`, `.mp3` or `.flac` file in here and pick a 30-second window with some contrast (a build-up,
a drop or a breakdown gives the visualizer the most to react to). `demo_track.wav` is re-created automatically by the notebook if it
is ever deleted.
