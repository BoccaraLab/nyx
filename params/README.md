# Parameter files

A parameter file describes **how to score**, independently of any particular
recording: the spectral settings, and the sequence of clustering steps. It
carries no paths, thresholds or cluster mappings — those belong to a single
recording and end up in that run's `run.json`.

So a parameter file is shareable. You can hand `mouse.json` to a colleague and
it means the same thing on their data.

## What ships here

| file | for |
|---|---|
| `mouse.json` | mouse EEG/EMG, EMG threshold then NREM/REM clustering |
| `mouse_weak_emg.json` | mouse, when the EMG does not separate wake cleanly, or the recording is short — one clustering step recovers all three stages using EMG as a feature |
| `rat.json` | rat EEG/EMG |
| `human.json` | human PSG, five stages, three clustering steps |

These are **starting points, not fixed recipes.** Run one recording, look at the
EMG threshold histogram and the per-cluster PSDs, and adjust. The defaults were
chosen to be reasonable, not optimal: `mouse.json` scores 0.94 against expert
consensus on the Gulledge benchmark with no tuning at all, but a few minutes of
per-recording attention will usually beat that.

## Structure

```jsonc
{
  "features": { "method": "spectrogram" },

  "EEG": { "min_freq": 0.5, "max_freq": 40, "binsize": 2, ... },
  "EMG": { "min_freq": 30,  "max_freq": 100, "binsize": 4, ... },

  "scoring": { "pc_components": 6, "min_duration": 4 },

  "steps": [ ... ]
}
```

`binsize` is the epoch length in seconds. `normalized` is one of `false`,
`"mean"`, `"zscore"` (per frequency bin across time — what makes components
comparable across subjects) or `"relative"`. `notch` is `null`, `50` or `60`;
run `nyx.check_signals()` and it will tell you which, if any, you need.

### Steps

Each step takes the epochs currently labelled `within`, fits a PCA inside just
those epochs, clusters them, and replaces that label with finer ones. Species
differ in how many steps they need, not in the code that runs them.

```jsonc
{ "name": "split_sleep",
  "within": "SLEEP",            // which label to subdivide
  "method": "hdbscan",          // hdbscan | kmeans | gmm | elliptic | emg_threshold
  "pcs_to_use": [0, 1, 2, 3],   // or "n_pcs": 4
  "use_emg": false,             // add EMG power as an extra feature
  "stage_order": ["REM", "NREM"] }   // names clusters by centroid, lowest first
```

`stage_order` names clusters by position, which is stable. `cluster_to_stage`
names them by id, which is stable too — nyx renumbers clusters by centroid so
that ids are a property of the data rather than of how the algorithm happened to
initialise. Mapping every cluster of a step to the same stage makes that step a
no-op, which is how you decide, after looking at it, that a split was not worth
keeping.

A step may also carry `refinements`, which divide one of its output stages by
thresholding a component:

```jsonc
"refinements": [
  { "split_stage": "NREM2", "pc": 0, "threshold": 1.19,
    "high": "NREM2", "low": "NREM1" }
]
```

### Comments

JSON has no comments, so any key beginning with `_` is ignored. Use `_comment`
to record *why* a value is what it is — that is usually the thing you will want
back in six months.

## Per-dataset parameters

The files here are per **species**. The exact parameters used for each dataset
in the paper — including per-recording thresholds and cluster mappings — live in
the analysis repository as `run.json` files, one per recording. Each of those is
self-contained and replayable: `nyx.load_config("run.json")` restores every
decision.
