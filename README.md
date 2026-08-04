# nyx

Unsupervised sleep scoring from EEG and EMG.

nyx separates wake from sleep using EMG power, then splits sleep into stages by
clustering a PCA of the EEG spectrogram. It needs **no training data and no
manually scored examples** — which is what lets it work on species that have no
scoring standard, and on recordings nobody has scored yet.

Rodent scoring (Wake/NREM/REM) and human scoring (5 stages) are the same code:
they differ in how many clustering steps run, which is configuration, not
different pipelines.

---

## Install

```bash
conda create -n nyx python=3.12 -y
conda activate nyx
pip install -e ".[notebooks]"
```

## Score a recording

```python
import nyx

recording = nyx.read_recording("mouse01.edf", eeg_channel=0, emg_channel=1)
print(recording.describe())         # check you picked the right channels

params = nyx.load_params("params/mouse.json")
result = nyx.score_recording(recording, params)
nyx.save_results(result, "results/mouse01")
```

That writes `hypnogram.csv` and a `run.json` recording every decision made.

A manually scored reference is **optional**. Pass one only if you have it and
want agreement metrics:

```python
reference = nyx.read_annotations("mouse01_scores.csv", format="epoch_csv",
                                 epoch_length=4,
                                 label_map={1: "WAKE", 2: "NREM", 3: "REM"})
result = nyx.score_recording(recording, params, reference=reference)
print(result.agreement.summary())
```

Start with [`examples/01_score_recording.ipynb`](examples/01_score_recording.ipynb),
which walks through a single recording step by step.

---

## Look at the signal first

Before scoring anything, check what you are working with:

```python
check = nyx.check_signals(recording)     # traces + spectrograms, start and end
check.plot()
print(check.summary())
```

This answers the three questions that come before scoring: is the signal usable,
does it need a notch filter, and where should the analysis window start and end.
Mains interference is measured rather than eyeballed —
`check.suggested_notch()` returns `50`, `60` or `None`.

## Three decisions are yours

nyx is not a black box, and three points in it are genuine judgement calls. Each
has an automatic default and an explicit override:

| decision | default | override |
|---|---|---|
| analysis window | whole recording | `window=(start, end)` |
| EMG wake/sleep threshold | fitted automatically | `emg_threshold=...` |
| which cluster is which stage | ordered by spectral content | `cluster_overrides={...}` |

Run one recording, look at the EMG histogram and the per-cluster PSDs, and
adjust. The defaults are reasonable, not optimal.

---

## Your own data

nyx reads recordings by **format**, not by dataset, so using it on new data
needs no code:

| recordings | annotations |
|---|---|
| `edf` / `bdf` | `interval_csv` — `time,duration,label` |
| `spikeinterface` folders | `epoch_csv` — one row per epoch, any stage coding |
| `npz` | `column_csv` — one column per recording |
| | `nsrr_xml`, `visbrain_hyp`, `epoch_npy`, `epoch_mat` |

Channels are selected by position **or** by name (`eeg_channel="C3_M2"`).
Anything unusual can be registered from your own code — see
`nyx.io.register_recording_reader` and `register_annotation_reader`.

### Many recordings

Batches are driven by a CSV, one row per recording:

```csv
recording_path,annotation_path,eeg_channel,emg_channel
data/m01.edf,scores/m01.csv,0,1
data/m02.edf,,0,1
```

```python
for config in nyx.load_manifest("recordings.csv", params="params/mouse.json"):
    recording, reference = config.load()
    result = nyx.score_recording(recording, config.params, reference=reference)
    nyx.save_results(result, config.output_dir)
```

### Public datasets

Recipes know the folder layout of several public datasets, so you only give a
root and a recording name:

```python
from nyx.datasets import list_datasets, load_dataset

recording, reference = load_dataset("dodh", data_root="/data",
                                    recording_name="0d79f4b1-...")
```

Currently: Oxford mouse benchmark, Gulledge 2025, Sippel, Ellen/Dash, SIESTA,
Boccara lab, Dreem Open Datasets (dodh/dodo), MESA, CHAT, CCSHS, ANPHY-Sleep.

---

## Parameters

A parameter file describes **how to score**, not which recording. It carries the
spectral settings and the sequence of clustering steps, and no paths or
thresholds — so it is shareable. See [`params/README.md`](params/README.md).

```jsonc
"steps": [
  { "name": "wake_sleep",  "method": "emg_threshold" },
  { "name": "split_sleep", "within": "SLEEP", "method": "hdbscan",
    "n_pcs": 4, "stage_order": ["REM", "NREM"] }
]
```

Shipped defaults: `mouse.json`, `mouse_weak_emg.json` (when the EMG does not
separate cleanly), `rat.json`, `human.json`.

## Reproducibility

Every run writes a `run.json` holding the parameters **inline** plus every
decision — window, thresholds, cluster mappings, refinement thresholds. Loading
it re-runs the scoring exactly, with nothing left to decide:

```python
config = nyx.load_config("results/mouse01/run.json")
```

Cluster ids are renumbered by spectral content rather than by whatever order the
clustering algorithm produced, so a saved mapping means the same thing on every
run. Without that, replaying saved parameters can attach stage names to the
wrong clusters.

---

## Tests

```bash
pytest
```

The suite runs on synthetic signals with a known hypnogram, so it needs no data.

## Citing

See `CITATION.cff`. The analyses in the accompanying paper live in a separate
repository, pinned to a specific nyx version.

## Licence

MIT.
