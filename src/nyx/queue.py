"""Working through a manifest one recording at a time, without losing your place.

A study is a list of recordings, and scoring it is the same few decisions
repeated -- the analysis window, the EMG threshold, which cluster is which
stage. There are two ways to get through that list, and they are for different
things.

**Headless.** Score everything with the parameters as written, no interaction::

    queue = nyx.open_queue("recordings.csv", params="params/mouse.json")
    queue.run_all()

This is how you *reproduce* a result: point it at a folder of ``run.json``
files and every decision is already recorded, so nothing is left to choose. It
is not how you should score a study for the first time -- default parameters on
unseen recordings are a starting point, not an answer.

**Interactive.** Work through the list in a notebook or a GUI, one recording at
a time::

    for config in nyx.open_queue("recordings.csv", params="params/mouse.json"):
        ...                    # score this one however you like
        queue.done()           # or queue.skip("EMG disconnected")

Each turn of the loop rewrites ``config.json`` to point at the next recording,
so the notebook stays untouched::

    import nyx
    config = nyx.load_config("config.json")
    recording, reference = config.load()

Change nothing in the notebook between recordings. Run it top to bottom, adjust
what needs adjusting, save the result, then ask the queue for the next one.

Picking up where you left off
-----------------------------

Progress is kept in a small JSON file beside the results, so an interrupted
session resumes rather than starting over. Even without it, a recording whose
output folder already holds a ``run.json`` counts as scored -- so the queue is
still right after the state file is deleted, or when someone else scored a few
recordings on another machine.

``skip`` is not the same as ``done``: a skipped recording is recorded *with its
reason*, so "why is there no result for m07" has an answer six months later.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from nyx.config import RunConfig
from nyx.manifest import load_manifest

__all__ = ["Queue", "QueueEntry", "open_queue"]

#: Where the queue writes the config for the recording being worked on. The
#: notebook reads this and nothing else, which is the point.
DEFAULT_CONFIG_PATH = "config.json"

#: Where progress is kept, relative to the output root.
STATE_FILENAME = "queue_state.json"

PENDING, DONE, SKIPPED, FAILED = "pending", "done", "skipped", "failed"


@dataclass
class QueueEntry:
    """One recording's place in the queue."""

    name: str
    status: str = PENDING
    note: str = ""
    at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "note": self.note, "at": self.at}


@dataclass
class Queue:
    """A manifest, plus what has been done to it so far.

    Build one with :func:`open_queue` rather than directly.
    """

    configs: list[RunConfig]
    output_root: str = "results"
    state_path: str = ""
    config_path: str = DEFAULT_CONFIG_PATH
    manifest_path: str = ""
    entries: dict[str, QueueEntry] = field(default_factory=dict)
    #: The recording handed out by the last :meth:`next`, if it is still open.
    current: RunConfig | None = None

    # -- what is left ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.configs)

    def entry(self, name: str) -> QueueEntry:
        return self.entries.setdefault(name, QueueEntry(name=name))

    def status_of(self, config: RunConfig) -> str:
        """This recording's status, trusting a written result over the state file.

        A recording whose output folder holds a ``run.json`` has been scored,
        whatever the state file says -- that keeps the queue honest when the
        state file is deleted, or when a colleague scored some of the list
        somewhere else.
        """
        recorded = self.entry(config.name).status
        if recorded in (SKIPPED, FAILED):
            return recorded
        if os.path.exists(os.path.join(config.output_dir, "run.json")):
            return DONE
        return recorded

    @property
    def pending(self) -> list[RunConfig]:
        return [c for c in self.configs if self.status_of(c) == PENDING]

    @property
    def scored(self) -> list[RunConfig]:
        """Recordings already scored. Named for the past tense of what happened,
        not ``done`` -- that is the verb you call to mark one."""
        return [c for c in self.configs if self.status_of(c) == DONE]

    @property
    def skipped(self) -> list[RunConfig]:
        return [c for c in self.configs if self.status_of(c) == SKIPPED]

    @property
    def failed(self) -> list[RunConfig]:
        return [c for c in self.configs if self.status_of(c) == FAILED]

    def summary(self) -> str:
        counts = {DONE: 0, PENDING: 0, SKIPPED: 0, FAILED: 0}
        for config in self.configs:
            counts[self.status_of(config)] += 1

        lines = [
            f"{self.manifest_path or 'queue'}: {len(self.configs)} recordings",
            f"  done {counts[DONE]}   pending {counts[PENDING]}"
            f"   skipped {counts[SKIPPED]}   failed {counts[FAILED]}",
        ]
        for config in self.configs:
            status = self.status_of(config)
            if status == PENDING:
                continue
            note = self.entry(config.name).note
            mark = {DONE: "ok", SKIPPED: "--", FAILED: "!!"}[status]
            lines.append(
                f"    {mark} {config.name}" + (f"  ({note})" if note else "")
            )
        if counts[PENDING]:
            nxt = self.pending[0].name
            lines.append(f"  next: {nxt}")
        return "\n".join(lines)

    def print_status(self) -> None:
        print(self.summary())

    # -- working through it ------------------------------------------------

    def next(self, write_config: bool = True) -> RunConfig | None:
        """Hand out the next unscored recording, or ``None`` when there are none.

        Writes :attr:`config_path` so the notebook picks this recording up
        without being edited. Call :meth:`done` or :meth:`skip` before asking
        for another.
        """
        remaining = self.pending
        if not remaining:
            self.current = None
            return None

        self.current = remaining[0]
        if write_config:
            self.write_config(self.current)
        return self.current

    def __iter__(self):
        """Yield pending recordings, writing the config for each.

        Marking is still yours: call :meth:`done` or :meth:`skip` in the body.
        A recording left unmarked stays pending, so an interrupted loop resumes
        on the one you were in the middle of rather than skipping past it.
        """
        while True:
            config = self.next()
            if config is None:
                return
            before = self.status_of(config)
            yield config
            if self.status_of(config) == before == PENDING:
                # Nothing was marked and nothing was written: stop rather than
                # hand out the same recording forever.
                raise RuntimeError(
                    f"{config.name} is still pending after its turn, so the queue "
                    f"would hand it out again indefinitely. Call queue.done() or "
                    f"queue.skip(reason) in the loop body, or "
                    f"queue.mark(name, 'done')."
                )

    def write_config(self, config: RunConfig) -> str:
        """Write one recording's config to :attr:`config_path`."""
        payload = config.to_dict()
        directory = os.path.dirname(os.path.abspath(self.config_path))
        os.makedirs(directory, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4, default=str)
        return self.config_path

    def mark(self, name: str, status: str, note: str = "") -> None:
        """Record a recording's outcome and save the state file."""
        if status not in (PENDING, DONE, SKIPPED, FAILED):
            raise ValueError(
                f"Unknown status {status!r}. Expected one of "
                f"{[PENDING, DONE, SKIPPED, FAILED]}."
            )
        entry = self.entry(name)
        entry.status = status
        entry.note = note
        entry.at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.save()

    def done(self, note: str = "") -> None:
        """Mark the current recording scored and move on."""
        self._mark_current(DONE, note)

    def skip(self, reason: str) -> None:
        """Mark the current recording deliberately not scored.

        The reason is kept, which is the difference between this and simply not
        getting to it: an unusable EMG or a recording that turned out to be the
        wrong animal should say so in the record rather than leaving a gap.
        """
        if not reason:
            raise ValueError(
                "Skipping needs a reason -- it is what tells you later why there "
                "is no result for this recording."
            )
        self._mark_current(SKIPPED, reason)

    def _mark_current(self, status: str, note: str) -> None:
        if self.current is None:
            raise RuntimeError(
                "No recording is open. Call queue.next() first, or "
                "queue.mark(name, status) to mark one by name."
            )
        self.mark(self.current.name, status, note)
        self.current = None

    def reset(self, name: str | None = None) -> None:
        """Put a recording (or the whole queue) back to pending.

        This clears the state file's record. A ``run.json`` already written
        still counts as done, so delete the output folder too if you mean to
        score it again from scratch.
        """
        if name is None:
            self.entries.clear()
        else:
            self.entries.pop(name, None)
        self.current = None
        self.save()

    # -- headless ----------------------------------------------------------

    def run_all(
        self,
        skip_scored: bool = True,
        stop_on_error: bool = False,
        verbose: bool = True,
        **score_kwargs,
    ) -> list[dict[str, Any]]:
        """Score every pending recording with no interaction.

        Use it to reproduce results -- point the manifest at saved ``run.json``
        files and every decision is already made. Scoring unseen recordings this
        way runs on defaults, which are a starting point rather than an answer;
        the interactive loop exists for that.

        Parameters
        ----------
        skip_scored
            Leave recordings that already have a ``run.json`` alone. Turn it off
            to re-score everything.
        stop_on_error
            Raise on the first failure. By default a failing recording is
            recorded as ``failed`` with its error and the batch carries on --
            one unreadable file should not cost you the other forty. Turn this
            on to get the traceback when you want to fix one.

        Returns
        -------
        list of dict
            One row per recording: ``name``, ``status``, ``mf1`` (when a
            reference was scored against), ``error``.
        """
        from nyx.pipeline import save_results, score_recording

        targets = self.pending if skip_scored else list(self.configs)
        rows: list[dict[str, Any]] = []

        for i, config in enumerate(targets, start=1):
            if verbose:
                print(f"[{i}/{len(targets)}] {config.name}", flush=True)

            row: dict[str, Any] = {"name": config.name, "status": DONE,
                                   "mf1": None, "error": ""}
            try:
                recording, reference = config.load()
                result = score_recording(
                    recording,
                    config.params,
                    window=config.window,
                    reference=reference,
                    verbose=False,
                    **score_kwargs,
                )
                save_results(result, config.output_dir, config=config)
                if result.agreement is not None:
                    row["mf1"] = round(result.agreement.mf1, 4)
                self.mark(config.name, DONE)
                if verbose:
                    got = "" if row["mf1"] is None else f"  MF1 {row['mf1']:.3f}"
                    print(f"    done{got}", flush=True)
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
                row["status"] = FAILED
                row["error"] = f"{type(exc).__name__}: {exc}"
                self.mark(config.name, FAILED, row["error"])
                if verbose:
                    # The message only: a batch of forty should not bury the
                    # recordings that worked under tracebacks. Re-run the one
                    # you care about with stop_on_error=True to get it.
                    print(f"    FAILED  {row['error']}", flush=True)
                if stop_on_error:
                    raise

            rows.append(row)

        return rows

    # -- persistence -------------------------------------------------------

    def save(self) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(self.state_path)), exist_ok=True)
        payload = {
            "manifest": self.manifest_path,
            "output_root": self.output_root,
            "entries": {name: entry.to_dict() for name, entry in self.entries.items()},
        }
        with open(self.state_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4)
        return self.state_path

    def load_state(self) -> None:
        if not os.path.exists(self.state_path):
            return
        with open(self.state_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        self.entries = {
            name: QueueEntry(name=name, **spec)
            for name, spec in payload.get("entries", {}).items()
        }


def open_queue(
    manifest: str,
    params: str | None = None,
    output_root: str = "results",
    config_path: str = DEFAULT_CONFIG_PATH,
    state_path: str | None = None,
    **defaults: Any,
) -> Queue:
    """Open a manifest as a resumable queue.

    Parameters
    ----------
    manifest
        The CSV, as :func:`nyx.load_manifest` reads it.
    params
        Parameter file for rows that do not name their own.
    output_root
        Where results go; each recording gets a subfolder.
    config_path
        Where the current recording's config is written. The notebook reads
        this file and is otherwise never edited.
    state_path
        Where progress is kept. Defaults to ``<output_root>/queue_state.json``.
    **defaults
        Passed to :func:`nyx.load_manifest` for blank columns.

    Examples
    --------
    Interactive, one recording at a time::

        queue = nyx.open_queue("recordings.csv", params="params/mouse.json")
        queue.print_status()

        config = queue.next()          # writes config.json
        # ... run the notebook, adjust, save ...
        queue.done()

    Headless, to reproduce a set of saved runs::

        nyx.open_queue("reruns.csv").run_all()
    """
    configs = load_manifest(manifest, params=params, output_root=output_root, **defaults)

    # Same resolution load_manifest applies, so the progress file lands beside
    # the results rather than somewhere else relative to the same argument.
    output_root = os.path.abspath(output_root)

    queue = Queue(
        configs=configs,
        output_root=output_root,
        state_path=state_path or os.path.join(output_root, STATE_FILENAME),
        config_path=config_path,
        manifest_path=manifest,
    )
    queue.load_state()
    return queue
