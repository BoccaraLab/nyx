"""The three configuration layers: params, config, and the batch manifest."""

from __future__ import annotations

import json

import pytest

import nyx
from nyx.config import RunConfig, load_config, load_params, validate_params
from nyx.manifest import load_manifest, write_manifest
from nyx.steps import Step, steps_from_params


def _write(tmp_path, name, payload):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload) if not isinstance(payload, str) else payload,
        encoding="utf-8",
    )
    return path


MINIMAL_PARAMS = {"EEG": {"binsize": 2}, "EMG": {"binsize": 4}}


# ---------------------------------------------------------------------------
# Params
# ---------------------------------------------------------------------------


def test_params_need_eeg_and_emg(tmp_path):
    path = _write(tmp_path, "p.json", {"EEG": {}})
    with pytest.raises(KeyError, match="EMG"):
        load_params(str(path))


def test_scalogram_params_need_a_frequency_resolution():
    """The two backends take different settings and are not interchangeable."""
    params = {**MINIMAL_PARAMS, "features": {"method": "scalogram"}}

    with pytest.raises(KeyError, match="freq_resolution"):
        validate_params(params)


def test_scalogram_params_declaring_the_spectrogram_are_caught_early():
    """Without this it fails deep in the feature code on a missing key, minutes
    into a long recording -- and the message points at the wrong thing."""
    params = {
        "EEG": {"freq_resolution": 1, "f0": 1, "exp_corr": 0},
        "EMG": {"binsize": 4},
    }
    with pytest.raises(KeyError, match="scalogram"):
        validate_params(params)


def test_a_valid_scalogram_params_file_passes():
    validate_params({
        "features": {"method": "scalogram"},
        "EEG": {"min_freq": 0.5, "max_freq": 30, "freq_resolution": 0.5},
        "EMG": {"min_freq": 30, "max_freq": 60, "freq_resolution": 1},
    })


def test_unknown_feature_method_is_rejected():
    with pytest.raises(ValueError, match="unknown features.method"):
        validate_params({**MINIMAL_PARAMS, "features": {"method": "wavelets"}})


def test_step_without_a_method_is_rejected():
    params = {**MINIMAL_PARAMS, "steps": [{"name": "s"}]}
    with pytest.raises(KeyError, match="no 'method'"):
        validate_params(params)


# ---------------------------------------------------------------------------
# Steps from JSON
# ---------------------------------------------------------------------------


def test_n_pcs_is_shorthand_for_pcs_to_use():
    step = Step.from_dict({"name": "s", "method": "kmeans", "n_pcs": 4})
    assert step.pcs_to_use == [0, 1, 2, 3]


def test_explicit_pcs_to_use_wins_over_n_pcs():
    step = Step.from_dict(
        {"name": "s", "method": "kmeans", "n_pcs": 4, "pcs_to_use": [0, 2]}
    )
    assert step.pcs_to_use == [0, 2]


def test_cluster_to_stage_keys_are_read_as_integers():
    """JSON object keys are strings; cluster ids are integers."""
    step = Step.from_dict(
        {"name": "s", "method": "kmeans", "cluster_to_stage": {"0": "REM", "1": "NREM"}}
    )
    assert step.cluster_to_stage == {0: "REM", 1: "NREM"}


def test_method_specific_settings_pass_through_as_options():
    step = Step.from_dict(
        {"name": "s", "method": "hdbscan", "hdbscan_min_cluster_size": 300}
    )
    assert step.options["hdbscan_min_cluster_size"] == 300
    assert step.clustering_params()["hdbscan_min_cluster_size"] == 300


def test_refinements_are_read_from_json():
    step = Step.from_dict(
        {
            "name": "s",
            "method": "kmeans",
            "refinements": [
                {"split_stage": "NREM2", "pc": 0, "threshold": 1.19,
                 "high": "NREM2", "low": "NREM1"}
            ],
        }
    )
    assert step.refinements[0].split_stage == "NREM2"
    assert step.refinements[0].threshold == 1.19


def test_step_roundtrips_through_json():
    original = Step.from_dict(
        {"name": "split_sleep", "within": "SLEEP", "method": "hdbscan",
         "pcs_to_use": [0, 1], "cluster_to_stage": {"0": "REM", "1": "NREM"},
         "hdbscan_min_samples": 30}
    )
    assert Step.from_dict(original.to_dict()) == original


def test_steps_are_read_in_order():
    params = {
        **MINIMAL_PARAMS,
        "steps": [
            {"name": "a", "method": "kmeans"},
            {"name": "b", "method": "hdbscan", "within": "WAKE"},
        ],
    }
    steps = steps_from_params(params)
    assert [s.name for s in steps] == ["a", "b"]
    assert steps[1].within == "WAKE"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_resolves_paths_relative_to_itself(tmp_path):
    _write(tmp_path, "params/mouse.json", MINIMAL_PARAMS)
    config_path = _write(tmp_path, "config.json", {
        "recording": {"path": "data/m01.edf", "eeg_channel": 0, "emg_channel": 1},
        "params": "params/mouse.json",
    })

    config = load_config(str(config_path))

    # A config can live next to its data and still be run from anywhere.
    assert config.recording.path == str(tmp_path / "data" / "m01.edf").replace("/", "\\") \
        or config.recording.path.endswith("m01.edf")
    assert config.params == MINIMAL_PARAMS


def test_config_needs_a_recording_or_a_dataset():
    with pytest.raises(KeyError, match="'recording' section"):
        RunConfig.from_dict({"params": {}})


def test_dataset_config_collects_recipe_arguments():
    config = RunConfig.from_dict({
        "dataset": "gulledge-2025",
        "data_root": "/data",
        "sub_dataset": "Saline (N=4)",
        "recording_name": "A31_D04",
        "params": MINIMAL_PARAMS,
    })

    assert config.dataset == "gulledge-2025"
    assert config.dataset_options["sub_dataset"] == "Saline (N=4)"
    assert config.name == "A31_D04"


def test_output_dir_is_named_after_the_recording():
    config = RunConfig.from_dict({
        "recording": {"path": "/data/m01.edf"},
        "output": {"root": "results"},
        "params": MINIMAL_PARAMS,
    })
    assert config.output_dir.replace("\\", "/") == "results/m01"


def test_annotations_are_optional():
    config = RunConfig.from_dict({
        "recording": {"path": "/data/m01.edf"}, "params": MINIMAL_PARAMS,
    })
    assert config.annotations is None


def test_annotation_reader_options_are_kept():
    config = RunConfig.from_dict({
        "recording": {"path": "/data/m01.edf"},
        "annotations": {"path": "/data/m01.csv", "format": "epoch_csv",
                        "epoch_length": 10, "label_map": {"1": "WAKE"}},
        "params": MINIMAL_PARAMS,
    })
    assert config.annotations.options["epoch_length"] == 10
    assert config.annotations.options["label_map"] == {"1": "WAKE"}


def test_config_roundtrips(tmp_path):
    spec = {
        "recording": {"path": "/data/m01.edf", "eeg_channel": "C3", "emg_channel": 1},
        "annotations": {"path": "/data/m01.csv", "format": "epoch_csv"},
        "params": MINIMAL_PARAMS,
        "output": {"root": "results", "granularities": [3]},
        "window": [0, 86000],
    }
    config = RunConfig.from_dict(spec)
    again = RunConfig.from_dict(config.to_dict())

    assert again.recording.path == config.recording.path
    assert again.recording.eeg_channel == "C3"
    assert again.window == (0.0, 86000.0)
    assert again.output.granularities == [3]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_minimal_manifest(tmp_path):
    _write(tmp_path, "recordings.csv",
           "recording_path,annotation_path\n"
           "data/m01.edf,scores/m01.csv\n"
           "data/m02.edf,\n")

    configs = load_manifest(str(tmp_path / "recordings.csv"))

    assert len(configs) == 2
    assert configs[0].annotations.path.endswith("m01.csv")
    # A blank annotation column means no reference scoring, which is fine.
    assert configs[1].annotations is None


def test_manifest_columns_override_defaults(tmp_path):
    _write(tmp_path, "m.csv",
           "recording_path,eeg_channel,emg_channel\n"
           "a.edf,C3_M2,EMG\n"
           "b.edf,,\n")

    configs = load_manifest(str(tmp_path / "m.csv"), eeg_channel=0, emg_channel=1)

    assert configs[0].recording.eeg_channel == "C3_M2"   # named channel
    assert configs[1].recording.eeg_channel == 0         # fell back to the default


def test_manifest_window_columns(tmp_path):
    _write(tmp_path, "m.csv",
           "recording_path,window_start,window_end\na.edf,100,5000\n")

    config = load_manifest(str(tmp_path / "m.csv"))[0]
    assert config.window == (100.0, 5000.0)


def test_manifest_can_use_dataset_recipes(tmp_path):
    _write(tmp_path, "m.csv",
           "dataset,recording_name,sub_dataset\n"
           "gulledge-2025,A31_D04,Saline (N=4)\n")

    config = load_manifest(str(tmp_path / "m.csv"), data_root="/data")[0]

    assert config.dataset == "gulledge-2025"
    assert config.dataset_options["recording_name"] == "A31_D04"
    assert config.dataset_options["sub_dataset"] == "Saline (N=4)"


def test_manifest_row_errors_name_the_line(tmp_path):
    _write(tmp_path, "m.csv", "recording_path,name\n,nameless\n")
    with pytest.raises(ValueError, match="line 2"):
        load_manifest(str(tmp_path / "m.csv"))


def test_manifest_skips_blank_lines(tmp_path):
    _write(tmp_path, "m.csv", "recording_path\na.edf\n\nb.edf\n")
    assert len(load_manifest(str(tmp_path / "m.csv"))) == 2


def test_empty_manifest_is_reported(tmp_path):
    _write(tmp_path, "m.csv", "recording_path\n")
    with pytest.raises(ValueError, match="no rows"):
        load_manifest(str(tmp_path / "m.csv"))


def test_write_manifest_only_emits_used_columns(tmp_path):
    path = tmp_path / "out.csv"
    write_manifest(str(path), [
        {"recording_path": "a.edf", "annotation_path": "a.csv"},
        {"recording_path": "b.edf", "annotation_path": "b.csv"},
    ])

    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "recording_path,annotation_path"


def test_written_manifest_reloads(tmp_path):
    path = tmp_path / "out.csv"
    write_manifest(str(path), [
        {"recording_path": "a.edf", "eeg_channel": "C3", "window_start": 0,
         "window_end": 1000},
    ])
    config = load_manifest(str(path))[0]

    assert config.recording.eeg_channel == "C3"
    assert config.window == (0.0, 1000.0)


# ---------------------------------------------------------------------------
# Built-in parameter presets
# ---------------------------------------------------------------------------


def test_every_shipped_preset_is_listed_and_loads():
    presets = nyx.available_params()

    assert "mouse" in presets and "human" in presets
    for name in presets:
        assert nyx.load_params(name)["EEG"]


def test_a_preset_loads_from_anywhere(tmp_path, monkeypatch):
    # The presets ship inside the package, so they must not depend on the
    # working directory the way "params/mouse.json" used to.
    monkeypatch.chdir(tmp_path)

    assert nyx.load_params("mouse")["EEG"]


def test_a_file_in_the_working_directory_wins_over_a_preset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path, "mouse.json", {**MINIMAL_PARAMS, "mine": True})

    assert nyx.load_params("mouse.json").get("mine") is True


def test_the_old_repo_root_path_still_works_but_warns():
    with pytest.warns(FutureWarning, match="built-in preset"):
        params = nyx.load_params("params/mouse.json")

    assert params["EEG"]


def test_a_name_that_is_neither_a_file_nor_a_preset_says_both():
    with pytest.raises(FileNotFoundError) as error:
        nyx.load_params("gerbil")

    assert "built-in presets" in str(error.value)
    assert "mouse" in str(error.value)


def test_a_config_can_name_a_preset_instead_of_a_path(tmp_path):
    path = _write(tmp_path, "config.json", {
        "recording": {"path": "rec.npz"},
        "params": "mouse",
    })

    config = nyx.load_config(str(path))

    assert config.params_path == "mouse"
    assert config.params["EEG"]
    # to_dict has to round-trip the name, not an absolute path, or the config
    # stops being portable the moment it is written back out.
    assert config.to_dict()["params"] == "mouse"
