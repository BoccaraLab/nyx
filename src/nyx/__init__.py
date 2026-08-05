"""nyx -- unsupervised sleep stage scoring from EEG and EMG.

nyx separates wake from sleep using EMG band power, then splits sleep into NREM
and REM by clustering a PCA of the EEG spectrogram. It needs no training data
and no manually scored examples.

Quick start::

    import nyx

    recording = nyx.read_recording("mouse01.edf", eeg_channel=0, emg_channel=1)
    print(recording.describe())          # check you picked the right channels

    params = nyx.load_params("params/mice.json")
    result = nyx.score_recording(recording, params)

    nyx.save_results(result, "results/mouse01")

A manually scored reference is **optional**. Pass one to compare against::

    reference = nyx.read_annotations("mouse01_scores.csv", epoch_length=4,
                                     label_map={1: "WAKE", 2: "NREM", 3: "REM"})
    result = nyx.score_recording(recording, params, reference=reference)
    print(result.agreement.summary())

To stop partway and adjust something, call the steps yourself -- see
:mod:`nyx.pipeline`.
"""

__version__ = "0.1.0"

from nyx.config import RunConfig, load_config, load_json, load_params
from nyx.demo import (
    demo_messy_recording,
    demo_params,
    demo_recording,
    demo_wideband_recording,
)
from nyx.emg_like import emg_from_lfp
from nyx.granularity import collapse
from nyx.inspect import SignalCheck, check_signals
from nyx.io import read_annotations, read_recording
from nyx.manifest import load_manifest, write_manifest
from nyx.pipeline import (
    ScoringResult,
    assign_stages,
    classify_wake_sleep,
    cluster_sleep,
    compute_emg_features,
    compute_sleep_pca,
    evaluate,
    find_wake_sleep_threshold,
    save_results,
    score_recording,
)
from nyx.postprocess import apply_rules
from nyx.preprocessing import preprocess_recording
from nyx.queue import Queue, open_queue
from nyx.report import plot_scoring_overview, plot_summary, save_report
from nyx.steps import Refinement, Step, run_step, run_steps
from nyx.types import (
    Agreement,
    EmgFeatures,
    Recording,
    SleepClusters,
    SleepPca,
    Staging,
    WakeSleep,
)

__all__ = [
    "__version__",
    # loading
    "read_recording",
    "read_annotations",
    "load_params",
    "load_json",
    # configuration
    "load_config",
    "RunConfig",
    "load_manifest",
    "write_manifest",
    "open_queue",
    "Queue",
    # multi-step scoring
    "Step",
    "Refinement",
    "run_step",
    "run_steps",
    "collapse",
    # try it without any data
    "demo_recording",
    "demo_messy_recording",
    "demo_wideband_recording",
    "demo_params",
    # figures
    "plot_summary",
    "save_report",
    "plot_scoring_overview",
    # step 0: look at the signal
    "check_signals",
    "SignalCheck",
    "preprocess_recording",
    "apply_rules",
    # pipeline steps
    "emg_from_lfp",
    "compute_emg_features",
    "find_wake_sleep_threshold",
    "classify_wake_sleep",
    "compute_sleep_pca",
    "cluster_sleep",
    "assign_stages",
    "evaluate",
    "score_recording",
    "save_results",
    # results
    "Recording",
    "EmgFeatures",
    "WakeSleep",
    "SleepPca",
    "SleepClusters",
    "Staging",
    "Agreement",
    "ScoringResult",
]
