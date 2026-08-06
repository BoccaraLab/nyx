"""Tab 4 -- the components, the clustering and the names, in one place.

The notebooks separate these into three STOPs, but they are one loop in
practice: you look at the per-cluster spectra, change ``min_cluster_size``,
look again, decide which cluster is REM, see the hypnogram, and change your
mind. Making that three screens would mean walking between them on every turn
of the loop.

They are still three different *costs*, and the two buttons say so:

* the PCA is minutes -- ``Recompute the PCA``;
* the clustering is seconds -- ``Recluster``, and it keeps the PCA;
* naming is free -- no button at all, it re-runs as you edit, because
  ``label_clusters`` keeps both the PCA and the clustering.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import nyx
from nyx.gui.canvas import FigureView, PanelCanvas
from nyx.gui.session import Stage
from nyx.gui.tabs.base import Tab, controls_column
from nyx.gui.widgets import ClusteringForm, PostprocessForm, StageTable
from nyx.stages import STAGE_ROW_ORDER

__all__ = ["SleepTab"]

#: Above this, PC1 is carrying nearly everything and the split that follows is
#: unlikely to mean much -- the notebooks say to go back to the signal.
_VARIANCE_WARNING = 0.90


class SleepTab(Tab):
    title = "Sleep stages"
    subtitle = (
        "Read the per-cluster spectra: NREM has more delta (1-4 Hz), REM has a "
        "theta peak around 6-9 Hz. Name the clusters in the table -- renaming "
        "never re-clusters, so it costs nothing to change your mind."
    )
    stage = Stage.STEPS
    run_label = "Recompute the PCA"

    def build(self) -> None:
        layout = QHBoxLayout(self.body)
        layout.setContentsMargins(0, 0, 0, 0)

        # -- components
        pca_box = QGroupBox("Components")
        pca_form = QFormLayout(pca_box)
        self.components = QSpinBox()
        self.components.setRange(2, 50)
        self.components.setValue(10)
        self.components.setToolTip(
            "How many components the PCA keeps. Changing this recomputes it."
        )
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

        # -- clustering
        self.clustering = ClusteringForm()
        recluster = QPushButton("Recluster")
        recluster.setToolTip("Keeps the PCA -- only the clustering is redone.")
        recluster.clicked.connect(self._recluster)

        # -- naming
        naming_box = QGroupBox("Names")
        naming_layout = QVBoxLayout(naming_box)
        naming_form = QFormLayout()
        self.stage_order = QLineEdit()
        self.stage_order.setPlaceholderText("REM, NREM")
        self.stage_order.setToolTip(
            "Stage names in cluster order, lowest centroid first. The table "
            "below overrides this per cluster."
        )
        self.stage_order.editingFinished.connect(self._rename)
        flip = QPushButton("Flip")
        flip.setToolTip("Reverse the order -- the usual fix when REM and NREM swap.")
        flip.clicked.connect(self._flip)

        order_row = QWidget()
        order_layout = QHBoxLayout(order_row)
        order_layout.setContentsMargins(0, 0, 0, 0)
        order_layout.addWidget(self.stage_order, 1)
        order_layout.addWidget(flip)
        naming_form.addRow("order", order_row)
        naming_layout.addLayout(naming_form)

        # -- postprocessing
        self.postprocess = PostprocessForm()
        self.postprocess.changed.connect(self._rename)

        self.use_postprocess = QCheckBox("apply the rules")
        self.use_postprocess.setToolTip(
            "Rules trade agreement for plausibility. They are off unless the "
            "params turn them on."
        )
        self.use_postprocess.toggled.connect(self._rename)

        layout.addWidget(
            controls_column(
                pca_box, self.clustering, recluster, naming_box,
                self.use_postprocess, self.postprocess,
            )
        )

        # -- figures and the table
        splitter = QSplitter()
        splitter.setOrientation(splitter.orientation().Vertical)

        self.figures = QTabWidget()
        self.pca_view = FigureView(placeholder="Run the PCA to see the components.")
        self.cluster_view = FigureView(
            placeholder="Cluster the epochs to see the scatter and the spectra."
        )
        self.feature_view = FigureView(
            placeholder="The dimensions the scatter does not show."
        )
        self.hypnogram = PanelCanvas(figsize=(9, 2.4))

        self.figures.addTab(self.cluster_view, "Clusters and spectra")
        self.figures.addTab(self.pca_view, "Components")
        self.figures.addTab(self.feature_view, "Other dimensions")
        self.figures.addTab(self.hypnogram, "Hypnogram")
        splitter.addWidget(self.figures)

        self.table = StageTable()
        self.table.changed.connect(self._rename)
        splitter.addWidget(self.table)
        splitter.setSizes([620, 240])

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(splitter)
        layout.addWidget(right, 1)

    # -- actions -----------------------------------------------------------

    def _recluster(self) -> None:
        self.session.set_clustering(self.clustering.values())
        self.run_requested.emit(Stage.STEPS)

    def _flip(self) -> None:
        names = _split(self.stage_order.text())
        self.stage_order.setText(", ".join(reversed(names)))
        self._rename()

    def _rename(self) -> None:
        """Rename and re-apply the rules. Instant -- nothing is re-clustered."""
        if not self.session.has(Stage.STEPS):
            return

        names = _split(self.stage_order.text())
        if names:
            self.session.set_stage_order(names)
        self.session.set_cluster_overrides(self.table.overrides() or None)
        self.session.set_postprocess(
            self.postprocess.rules() if self.use_postprocess.isChecked() else []
        )
        self.session.compute(Stage.STEPS)
        self._draw_hypnogram()
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
            if not self.stage_order.text().strip():
                order = step.stage_order or ["REM", "NREM"]
                with self.quiet(self.stage_order):
                    self.stage_order.setText(", ".join(order))

        rules = self.session.postprocess_rules()
        with self.quiet(self.use_postprocess):
            self.use_postprocess.setChecked(bool(rules))
        self.postprocess.set_rules(rules)

        if not self.session.has(Stage.STEPS):
            return

        self._redraw_figures()
        self._draw_hypnogram()
        self._fill_table()

    # -- drawing -----------------------------------------------------------

    def _redraw_figures(self) -> None:
        if not self.session.has(Stage.STEPS):
            return
        pca = self.session.pca()
        clusters = self.session.clusters()
        stages = self.session.cluster_to_stage()
        smooth = float(self.smooth.value())

        self.pca_view.set_figure(nyx.plot_pca_grid(pca, smooth=smooth))
        self.cluster_view.set_figure(
            nyx.plot_cluster_check(pca, clusters=clusters, stages=stages, smooth=smooth)
        )
        self.feature_view.set_figure(
            nyx.plot_cluster_features(clusters, stages=stages)
        )

        explained = pca.explained_variance_ratio
        first = float(explained[0]) if len(explained) else 0.0
        text = "  ".join(f"PC{i + 1} {100 * v:.0f}%" for i, v in enumerate(explained[:4]))
        if first > _VARIANCE_WARNING:
            text += (
                f"\n\nPC1 carries {100 * first:.0f}% of the variance. That "
                f"usually means the spectrogram is dominated by something "
                f"other than sleep -- go back and check the signal."
            )
        self.variance.setText(text)

    def _draw_hypnogram(self) -> None:
        try:
            staging = self.session.staging()
        except Exception:  # noqa: BLE001
            return
        self.hypnogram.draw_panel(nyx.plot_hypnogram_result, staging)

    def _fill_table(self) -> None:
        try:
            frame = self.session.cluster_table()
        except Exception:  # noqa: BLE001
            return
        with self.quiet(self.table):
            self.table.set_table(frame, _stage_choices(self.session))


def _split(text: str) -> list[str]:
    return [part.strip() for part in str(text).replace(",", " ").split() if part.strip()]


def _stage_choices(session) -> list[str]:
    """Stage names to offer, the vocabulary first and anything else after."""
    names = list(STAGE_ROW_ORDER)
    for step in session.steps:
        for name in step.stage_order or []:
            if name not in names:
                names.append(name)
    return names
