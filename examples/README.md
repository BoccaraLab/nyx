# Examples

Work through them in order. The first two need **no data at all** — they run
on a synthetic recording, so you can see how everything fits together before
pointing nyx at your own files.

| notebook | needs | what it covers |
|---|---|---|
| [`01_score_recording.ipynb`](01_score_recording.ipynb) | nothing | one recording, step by step, and the four decisions that are yours |
| [`02_signal_check.ipynb`](02_signal_check.ipynb) | nothing | looking at the signal first: mains interference, artefacts, and what fixing them is worth |
| [`04_human_five_stage.ipynb`](04_human_five_stage.ipynb) | a human PSG recording | the human pipeline start to end: four steps, five stages, and collapsing to coarser ones |
| [`03_score_rodents.ipynb`](03_score_rodents.ipynb) | one 78 MB download | the rodent pipeline start to end, on a real mouse recording scored against a ten-expert consensus |

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
measure of how well nyx works**. Notebook 03 is the honest one: a real recording,
a real reference, and defaults that will need adjusting.

## Where the data goes

Notebook 03 downloads into `examples/data/`, which is git-ignored. Delete it when
you are done; nothing else depends on it.
