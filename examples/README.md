# Examples

Work through them in order. The first two need **no data at all** — they run
on a synthetic recording, so you can see how everything fits together before
pointing nyx at your own files.

| notebook | needs | summary |
|---|---|---|
| [`01_score_recording.ipynb`](01_score_recording.ipynb) | nothing | one recording, step by step, and the four decision points |
| [`02_signal_check.ipynb`](02_signal_check.ipynb) | nothing | looking at the signal first: mains interference, artefacts, and what fixing them is worth |
| [`03_score_rodents.ipynb`](03_score_rodents.ipynb) | one recorded downloaded in the notebook | the rodent pipeline start to end, on a real mouse recording scored against a ten-expert consensus |
| [`04_score_humans.ipynb`](04_score_humans.ipynb) | a human PSG recording (check open source human repos in the last section) | the human pipeline start to end: four steps, five stages, and collapsing to coarser ones |

## Or use the GUI

If you would rather click than type:

```bash
pip install -e ".[gui]"
nyx-gui
```

It has a tab per decision these notebooks stop at, and a **Try the demo
recording** button on the first one, so it needs no data either.

What it adds over a notebook is that the views are live: the traces,
spectrograms and hypnograms scroll together and the panels can be dragged
around, the EMG threshold is a line you drag rather than a number you retype,
and the hypnogram can be corrected epoch by epoch. It calls the same functions,
so anything you work out in one transfers to the other.

There is also [`run_demo.py`](run_demo.py), which needs nothing and no notebook
server:

```bash
python examples/run_demo.py
```

It scores a synthetic recording, then repeats the exercise on one carrying mains
interference and saturated signal at both ends — showing the signal check
finding both problems and what fixing them is worth. Every figure lands in
`examples/demo_output/`.

## Running them

```bash
pip install -e ".[notebooks]"
jupyter lab
```

## A caution about the synthetic data

The synthetic signal in notebooks 01–02 is built to contain exactly the
structure nyx looks for, so it scores near-perfectly. That makes it good for
learning the mechanics and for checking an installation, and **useless as a
measure of how well nyx works**.

## Where the data goes

Notebook 03 downloads into `examples/data/`, which is git-ignored.

## Public datasets

nyx ships a **recipe** for each of these: it knows the folder layout and the
scoring format, so you give a root and a recording name and nothing else.

```python
from nyx.datasets import list_datasets, load_dataset

list_datasets()
recording, reference = load_dataset("dodh", data_root="/data",
                                    recording_name="...")
```

Getting the data is up to you — all of it comes from its original source, some
of it behind a free application. Download it, keep the layout the recipe
expects, and point `data_root` at the folder holding the dataset directories.

| recipe | dataset | where to get it |
|---|---|---|
| `dodh`, `dodo` | Dreem Open Datasets (healthy, apnoea) | [zenodo.org/records/15900394](https://zenodo.org/records/15900394) |
| `mesa`, `mesa_aged` | MESA | [sleepdata.org](https://sleepdata.org) (free application) |
| `ccshs` | CCSHS | [sleepdata.org](https://sleepdata.org) (free application) |
| `chat` | CHAT | [sleepdata.org](https://sleepdata.org) (free application) |
| `gulledge-2025` | Gulledge 2025 | [sleepdata.org/datasets/gulledge-2025](https://sleepdata.org/datasets/gulledge-2025) |
| `anphy_sleep_human` | ANPHY-Sleep | [osf.io/r26fh](https://osf.io/r26fh/overview) |
| `oxford_mouse_benchmark_dataset` | Oxford mouse polysomnography benchmark | [zenodo.org/records/10200482](https://zenodo.org/records/10200482) |
| `ellen_dash` | Ellen / Dash | [zenodo.org/records/5227351](https://zenodo.org/records/5227351) |
| `siesta_DLI` | SIESTA | [zenodo.org/records/15322394](https://zenodo.org/records/15322394) |
| `sippel_morris_water_maze` | Sippel — Morris water maze | [EBRAINS](https://search.kg.ebrains.eu/instances/Dataset/830f33f1-cb6e-45ac-9203-11f23e97c273) |
