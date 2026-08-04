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
comparable across subjects) or `"relative"`.

### Preprocessing

Two optional settings per channel, applied through spikeinterface before
anything else — lazily, and once, so the signal check and the scoring see the
same signal:

| setting | effect |
|---|---|
| `notch` | mains frequency to remove: `50` (Europe) or `60` (Americas), plus harmonics. `null` disables it. Run `nyx.check_signals()` and it will tell you which, if any, you need. |
| `resample` | target rate in Hz. Lowering it speeds everything up; keep it above twice your highest frequency of interest. `null` leaves the rate alone. |

Both are per channel, because notching the EMG but not the EEG is a normal
thing to want.

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

### Postprocessing

Clustering scores each epoch on its own, so it has no notion of what a
plausible *sequence* looks like. An optional `postprocess` list adds that back,
as an ordered list in the same shape as `preprocess`:

```jsonc
"postprocess": [
  { "rem_after_wake": { "min_wake_duration": 0 } },
  { "min_duration":   { "seconds": 4 } }
]
```

| rule | effect |
|---|---|
| `rem_flanked_by_wake` | REM with WAKE on both sides becomes WAKE. `max_duration` limits it to short bouts. |
| `rem_after_wake` | REM *following* a WAKE bout becomes WAKE. `min_wake_duration` limits it to REM after a long enough wake bout. |
| `min_duration` | No segment shorter than `seconds` survives. 4 for rodents, 30 for human recordings. |

**Off by default, and worth leaving off unless you know which readout you are
protecting.** Global agreement barely moves — in the paper's comparison Cohen's
kappa sat at ~0.83 for every rule, including no rule at all. What moves is the
biology: `rem_after_wake` pulls the REM/NREM ratio towards the manual value and
pushes total sleep time away from it, and `min_duration` cuts the microarousal
count by about a sixth but overshoots. Whichever you choose, `run.json` records
it — say in your methods which one you used.

The five rules compared in the paper are `rem_after_wake` with
`min_wake_duration` 20 (R3) or 0 (R4), `min_duration` at 4 s (R5), and the two
orderings of R4 and R5 (R6, R7). Order matters and is the order you write.

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
