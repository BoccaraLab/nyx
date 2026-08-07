"""Tab 4 -- the components, the clustering and the names, in one place.

The notebooks separate these into three STOPs, but they are one loop: you read
the per-cluster spectra, change ``min_cluster_size``, read them again, decide
which cluster is REM, look at the hypnogram, and change your mind. Three
screens would mean walking between them on every turn of that loop.

They are still three different *costs*, and the buttons say so: the PCA is
minutes, reclustering is seconds and keeps the PCA, and naming is free -- so
it has no button and re-runs as you edit.

The panels are dockable. The cluster scatter and the spectra are matplotlib,
because they are not time series and pyqtgraph would draw them worse; the
component scores and the hypnogram are real viewers, so they scroll against
each other and against the traces. Drag them wherever you want them -- side by
side is usually right for the scatter and the spectra.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import nyx
from nyx.gui.panels import MplPanel
from nyx.gui.session import Stage
from nyx.gui.tabs.base import Tab
from nyx.gui.widgets import ClusteringForm, PostprocessForm, StageTable
from nyx.stages import STAGE_ROW_ORDER

__all__ = ["SleepTab"]

#: Above this, PC1 carries nearly everything and the split that follows is
#: unlikely to mean much -- the notebooks say to go back to the signal.
VARIANCE_WARNING = 0.90


class SleepTab(Tab):
    title = "Sleep stages"
    subtitle = (
        "Read the per-cluster spectra: NREM has more delta (1-4 Hz), REM a "
        "theta peak around 6-9 Hz. Name them in the table -- renaming never "
        "re-clusters, so changing your mind is free."
    )
    run_at_top = True
    stage = Stage.STEPS
    run_label = "Recompute the PCA"
    controls_width = 360

    def build_controls(self) -> list:
        self._built_for = None

        pca_box = QGroupBox("Components")
        pca_form = QFormLayout(pca_box)
        self.components = QSpinBox()
        self.components.setRange(2, 50)
        self.components.setValue(10)
        self.components.setToolTip("Changing this recomputes the PCA.")
        pca_form.addRow("computed", self.components)

        self.smooth = QDoubleSpinBox()
        self.smooth.setRange(0.0, 100.0)
        self.smooth.setDecimals(0)
        self.smooth.setToolTip(
            "Display only -- smooths the drawn spectra, never the data they "
            "were computed from."
        )
        self.smooth.valueChanged.connect(self._redraw_figures)
        pca_form.addRow("smoothing", self.smooth)

        self.variance = QLabel()
        self.variance.setWordWrap(True)
        self.variance.setStyleSheet("color: palette(mid);")
        pca_form.addRow("", self.variance)

        self.clustering = ClusteringForm()
        recluster = QPushButton("Recluster")
        recluster.setToolTip("Keeps the PCA -- only the clustering is redone.")
        recluster.setMinimumHeight(28)
        recluster.clicked.connect(self._recluster)

        naming_box = QGroupBox("Names")
        naming_layout = QVBoxLayout(naming_box)
        naming_note = QLabel(
            "One row per cluster, in centroid order. Name them here -- "
            "renaming never re-clusters, so changing your mind is free."
        )
        naming_note.setWordWrap(True)
        naming_note.setStyleSheet("color: palette(mid);")
        naming_layout.addWidget(naming_note)

        self.table = StageTable()
        self.table.changed.connect(self._rename)
        self.table.setMinimumHeight(160)
        naming_layout.addWidget(self.table)

        manual_box = QGroupBox("Assign by hand")
        manual_layout = QVBoxLayout(manual_box)
        manual_note = QLabel(
            "Draw round a group of points in the scatter and give it a stage. "
            "For the cases clustering will not get on its own -- a REM cluster "
            "that merged into NREM, say."
        )
        manual_note.setWordWrap(True)
        manual_note.setStyleSheet("color: palette(mid);")
        manual_layout.addWidget(manual_note)

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self.lasso_stage = QComboBox()
        self.lasso_stage.addItems(list(STAGE_ROW_ORDER))
        self.lasso_stage.setCurrentText("REM")
        row_layout.addWidget(self.lasso_stage, 1)
        self.lasso = QPushButton("Lasso")
        self.lasso.setCheckable(True)
        self.lasso.setToolTip(
            "Click points in the scatter to trace a polygon; press enter or "
            "space to close it. Escape starts over."
        )
        self.lasso.toggled.connect(self._toggle_lasso)
        row_layout.addWidget(self.lasso)
        manual_layout.addWidget(row)

        clear_manual = QPushButton("Clear hand assignments")
        clear_manual.clicked.connect(self._clear_manual)
        manual_layout.addWidget(clear_manual)

        self.manual_note = QLabel()
        self.manual_note.setWordWrap(True)
        self.manual_note.setStyleSheet("color: palette(mid);")
        manual_layout.addWidget(self.manual_note)

        self.use_postprocess = QCheckBox("apply the postprocessing rules")
        self.use_postprocess.setToolTip(
            "Rules trade agreement for plausibility. They are off unless the "
            "params turn them on."
        )
        self.use_postprocess.toggled.connect(self._rename)
        self.postprocess = PostprocessForm()
        self.postprocess.changed.connect(self._rename)

        return [
            pca_box, self.clustering, recluster, naming_box, manual_box,
            self.use_postprocess, self.postprocess,
        ]

    def build_docks(self) -> None:
        self.clusters_panel = MplPanel(
            "clusters and spectra",
            placeholder="Run the PCA and clustering to see the scatter and "
                        "the per-cluster spectra.",
        )
        self.features_panel = MplPanel(
            "other dimensions",
            placeholder="The dimensions the scatter does not show.",
        )
        self.pca_panel = MplPanel(
            "component spectra", placeholder="What each component weighs."
        )

    # -- panels ------------------------------------------------------------

    def _build_viewers(self) -> None:
        """Rebuild the dock area when the underlying data changes.

        Keyed on the clustering, so renaming -- which changes only the labels
        -- swaps the hypnogram in place and leaves the scroll position alone.
        """
        if not self.session.has(Stage.STEPS):
            return
        key = (id(self.session.pca()), id(self.session.clusters()))
        if self._built_for == key:
            return

        self.docks.clear()
        self._built_for = key

        # Clustering panels only. The traces and the hypnogram belong to the
        # tabs either side of this one; here they are just competing for width
        # with the scatter you are actually reading.
        self.docks.add(self.clusters_panel)
        self.docks.add(self.features_panel, tabify_with="clusters and spectra")
        self.docks.add(self.pca_panel, tabify_with="other dimensions")

    # -- assigning by hand -------------------------------------------------

    def _toggle_lasso(self, on: bool) -> None:
        """Draw a polygon on the cluster scatter and name what falls inside.

        Uses the same :class:`~nyx.interactive.PolygonSelector` the notebooks
        use, on the same axes -- so what you learn in one place works in the
        other.
        """
        if not on:
            self._finish_lasso()
            return

        if not self.session.has(Stage.STEPS):
            self.lasso.setChecked(False)
            return

        figure = self.clusters_panel.view.figure
        if figure is None or not figure.axes:
            self.lasso.setChecked(False)
            self.status.emit("Cluster the epochs first.")
            return

        from nyx.interactive import PolygonSelector

        clusters = self.session.clusters()
        scatter_ax = figure.axes[0]          # plot_cluster_check draws it left
        points = clusters.features_scaled[:, :2]

        self._selector = PolygonSelector(scatter_ax, points, verbose=False)
        self.manual_note.setText(
            "Click to place vertices, enter or space to close, escape to "
            "start over. Then press Lasso again."
        )
        self.status.emit("Draw round the points you want.")

    def _finish_lasso(self) -> None:
        selector = getattr(self, "_selector", None)
        if selector is None:
            return
        if not selector.finished:
            selector.finish()

        mask = selector.get_mask()
        self._selector = None
        if not mask.any():
            self.manual_note.setText("Nothing was selected.")
            return

        stage = self.lasso_stage.currentText()
        count = self.session.assign_epochs(mask, stage)
        self.manual_note.setText(
            f"{count} epochs assigned to {stage} by hand."
        )
        self.status.emit(f"{count} epochs -> {stage}.")
        self._redraw_figures()
        self._fill_table()

    def _clear_manual(self) -> None:
        self.session.clear_manual_epochs()
        self.manual_note.setText("")
        self._redraw_figures()
        self._fill_table()
        self.status.emit("Hand assignments cleared.")

    # -- actions -----------------------------------------------------------

    def _recluster(self) -> None:
        """Recluster with the current settings, forgetting the old names.

        Cluster ids only mean anything relative to a particular clustering, so
        carrying a mapping across a change in the number of clusters is how you
        end up naming a cluster that no longer exists.
        """
        self.session.set_cluster_overrides(None)
        self.session.set_clustering(self.clustering.values())
        self.run_requested.emit(Stage.STEPS)

    def _rename(self) -> None:
        """Rename and re-apply the rules. Instant -- nothing is re-clustered.

        The table is the only thing that names clusters. It has exactly one row
        per cluster, so it cannot get out of step with the clustering the way a
        typed list of names could -- which is what used to break when the
        cluster count changed and the names had not been decided yet.
        """
        if not self.session.has(Stage.STEPS):
            return

        try:
            self.session.set_cluster_overrides(self.table.overrides() or None)
            self.session.set_postprocess(
                self.postprocess.rules() if self.use_postprocess.isChecked() else []
            )
            self.session.compute(Stage.STEPS)
        except Exception as exc:  # noqa: BLE001 - reported, never fatal
            self.show_warnings([f"Could not apply those names: {exc}"])
            return

        self._redraw_figures()
        self._fill_table()

    # -- session -----------------------------------------------------------

    def apply(self) -> None:
        params = dict(self.session.params)
        scoring = dict(params.get("scoring", {}))
        scoring["pc_components"] = int(self.components.value())
        params["scoring"] = scoring
        self.session.set_params(params)
        self.session.set_clustering(self.clustering.values())

    def refresh(self) -> None:
        from nyx.pipeline import resolve_clustering_params

        with self.quiet(self.components):
            self.components.setValue(
                int(self.session.params.get("scoring", {}).get("pc_components", 10))
            )

        step = self.session.steps[-1] if self.session.steps else None
        if step is not None:
            with self.quiet(self.clustering):
                self.clustering.set_values(
                    {**resolve_clustering_params(self.session.params),
                     **step.clustering_params()}
                )

        rules = self.session.postprocess_rules()
        with self.quiet(self.use_postprocess):
            self.use_postprocess.setChecked(bool(rules))
        self.postprocess.set_rules(rules)

        if not self.session.has(Stage.STEPS):
            return

        self._build_viewers()
        self._redraw_figures()
        self._fill_table()

    # -- drawing -----------------------------------------------------------

    def _redraw_figures(self) -> None:
        if not self.session.has(Stage.STEPS):
            return
        pca = self.session.pca()
        clusters = self.session.clusters()
        stages = self.session.cluster_to_stage()
        smooth = float(self.smooth.value())

        self.clusters_panel.set_figure(
            nyx.plot_cluster_check(pca, clusters=clusters, stages=stages,
                                   smooth=smooth)
        )
        self.features_panel.set_figure(
            nyx.plot_cluster_features(clusters, stages=stages)
        )
        self.pca_panel.set_figure(nyx.plot_pca_grid(pca, smooth=smooth))

        explained = pca.explained_variance_ratio
        first = float(explained[0]) if len(explained) else 0.0
        text = "  ".join(
            f"PC{i + 1} {100 * v:.0f}%" for i, v in enumerate(explained[:4])
        )
        if first > VARIANCE_WARNING:
            text += (
                f"\n\nPC1 carries {100 * first:.0f}% of the variance. That "
                f"usually means the spectrogram is dominated by something "
                f"other than sleep -- go back and check the signal."
            )
        self.variance.setText(text)

    def _fill_table(self) -> None:
        """One row per cluster, pre-filled with a plausible name.

        A freshly reclustered run has no mapping yet, so the rows are seeded
        from the step's declared stage order by position -- and any cluster
        beyond that order is left UNCLASSIFIED rather than silently folded into
        the last stage named.
        """
        try:
            frame = self.session.cluster_table()
        except Exception:  # noqa: BLE001 - nothing to show yet
            return

        step = self.session.steps[-1] if self.session.steps else None
        defaults = list((step.stage_order if step else None) or ["REM", "NREM"])
        with self.quiet(self.table):
            self.table.set_table(
                frame, _stage_choices(self.session), defaults=defaults
            )


def _stage_choices(session) -> list[str]:
    names = list(STAGE_ROW_ORDER)
    for step in session.steps:
        for name in step.stage_order or []:
            if name not in names:
                names.append(name)
    return names
