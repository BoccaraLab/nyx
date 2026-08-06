"""Deciding by pointing: the polygon lasso and the draggable threshold.

Both classes are pure matplotlib, so they are driven here the way matplotlib
itself drives them -- by pushing events through the canvas callback registry.
No Qt, no display.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseButton, MouseEvent

from nyx.interactive import PolygonSelector, ThresholdSelector


def click(ax, x, y, button=MouseButton.LEFT, kind="button_press_event"):
    """Synthesise a mouse event at *data* coordinates ``(x, y)``."""
    px, py = ax.transData.transform((x, y))
    event = MouseEvent(kind, ax.figure.canvas, px, py, button)
    ax.figure.canvas.callbacks.process(kind, event)
    return event


def press_key(ax, key):
    event = KeyEvent("key_press_event", ax.figure.canvas, key)
    ax.figure.canvas.callbacks.process("key_press_event", event)


@pytest.fixture
def figure():
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


# ---------------------------------------------------------------------------
# PolygonSelector
# ---------------------------------------------------------------------------


@pytest.fixture
def points():
    # A tight blob near the origin and another far away, so a polygon round
    # the first has an unambiguous right answer.
    near = np.array([[0.0, 0.0], [0.1, 0.1], [0.1, -0.1], [-0.1, 0.05]])
    far = np.array([[5.0, 5.0], [5.2, 4.8]])
    return np.vstack([near, far])


def test_a_polygon_selects_the_points_inside_it(figure, points):
    fig, ax = figure
    ax.scatter(points[:, 0], points[:, 1])
    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    selector = PolygonSelector(ax, points, verbose=False)

    for x, y in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]:
        click(ax, x, y)
    selector.finish()

    assert list(selector.get_mask()) == [True, True, True, True, False, False]


def test_an_abandoned_polygon_selects_nothing(figure, points):
    fig, ax = figure
    selector = PolygonSelector(ax, points, verbose=False)

    click(ax, 0.0, 0.0)
    click(ax, 1.0, 0.0)

    assert not selector.get_mask().any()
    assert not selector.finished  # two vertices is not a polygon


def test_a_right_click_takes_back_the_last_vertex(figure, points):
    fig, ax = figure
    selector = PolygonSelector(ax, points, verbose=False)

    click(ax, 0.0, 0.0)
    click(ax, 1.0, 1.0)
    click(ax, 2.0, 2.0, button=MouseButton.RIGHT)

    assert len(selector.polygon_points) == 1


def test_escape_starts_the_polygon_over(figure, points):
    fig, ax = figure
    selector = PolygonSelector(ax, points, verbose=False)

    for x, y in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5)]:
        click(ax, x, y)
    press_key(ax, "escape")

    assert selector.polygon_points == []


def test_points_must_be_two_dimensional(figure):
    fig, ax = figure

    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        PolygonSelector(ax, np.zeros((10, 3)))


# ---------------------------------------------------------------------------
# ThresholdSelector
# ---------------------------------------------------------------------------


@pytest.fixture
def panels():
    """The two halves of plot_emg_check: a distribution and a time course."""
    fig, (hist_ax, trace_ax) = plt.subplots(1, 2)
    hist_ax.set_xlim(0, 1)
    trace_ax.set_ylim(0, 1)
    yield fig, hist_ax, trace_ax
    plt.close(fig)


def line_positions(selector):
    """Where each line actually sits, read back off the artists."""
    out = []
    for _ax, line, orientation in selector._lines:
        data = line.get_xdata() if orientation == "v" else line.get_ydata()
        out.append(float(np.asarray(data)[0]))
    return out


def test_both_views_start_at_the_same_value(panels):
    _fig, hist_ax, trace_ax = panels

    selector = ThresholdSelector(0.4, hist_ax=hist_ax, trace_ax=trace_ax)

    assert selector.value == pytest.approx(0.4)
    assert line_positions(selector) == pytest.approx([0.4, 0.4])


def test_dragging_the_histogram_line_moves_the_trace_line_too(panels):
    _fig, hist_ax, trace_ax = panels
    selector = ThresholdSelector(0.4, hist_ax=hist_ax, trace_ax=trace_ax)

    click(hist_ax, 0.4, 0.5)  # grab it where it is
    click(hist_ax, 0.7, 0.5, kind="motion_notify_event")
    click(hist_ax, 0.7, 0.5, kind="button_release_event")

    assert selector.value == pytest.approx(0.7)
    assert line_positions(selector) == pytest.approx([0.7, 0.7])


def test_dragging_the_trace_line_moves_the_histogram_line_too(panels):
    _fig, hist_ax, trace_ax = panels
    selector = ThresholdSelector(0.4, hist_ax=hist_ax, trace_ax=trace_ax)

    # The trace is horizontal, so the value is on its y axis.
    click(trace_ax, 0.5, 0.4)
    click(trace_ax, 0.5, 0.2, kind="motion_notify_event")
    click(trace_ax, 0.5, 0.2, kind="button_release_event")

    assert selector.value == pytest.approx(0.2)
    assert line_positions(selector) == pytest.approx([0.2, 0.2])


def test_on_change_fires_per_motion_and_on_release_fires_once(panels):
    _fig, hist_ax, trace_ax = panels
    changes, releases = [], []
    selector = ThresholdSelector(
        0.4, hist_ax=hist_ax, trace_ax=trace_ax,
        on_change=changes.append, on_release=releases.append,
    )

    click(hist_ax, 0.4, 0.5)
    for x in (0.5, 0.6, 0.7):
        click(hist_ax, x, 0.5, kind="motion_notify_event")
    click(hist_ax, 0.7, 0.5, kind="button_release_event")

    # This split is the whole API: on_change is cheap and continuous,
    # on_release is where classify_wake_sleep goes.
    assert changes == pytest.approx([0.5, 0.6, 0.7])
    assert releases == pytest.approx([0.7])
    assert selector.value == pytest.approx(0.7)


def test_a_click_away_from_the_line_does_not_grab_it(panels):
    _fig, hist_ax, trace_ax = panels
    selector = ThresholdSelector(0.4, hist_ax=hist_ax, trace_ax=trace_ax)

    click(hist_ax, 0.9, 0.5)  # far from the line
    click(hist_ax, 0.1, 0.5, kind="motion_notify_event")

    assert selector.value == pytest.approx(0.4)


def test_the_value_is_clamped_to_the_limits(panels):
    _fig, hist_ax, trace_ax = panels
    selector = ThresholdSelector(
        0.4, hist_ax=hist_ax, trace_ax=trace_ax, limits=(0.2, 0.8)
    )

    selector.set_value(5.0)
    assert selector.value == pytest.approx(0.8)

    selector.set_value(-5.0)
    assert selector.value == pytest.approx(0.2)


def test_limits_default_to_the_histogram_axis(panels):
    _fig, hist_ax, trace_ax = panels
    hist_ax.set_xlim(0, 2)

    selector = ThresholdSelector(0.4, hist_ax=hist_ax, trace_ax=trace_ax)

    assert selector.limits == pytest.approx((0.0, 2.0))


def test_setting_the_value_quietly_notifies_nobody(panels):
    _fig, hist_ax, trace_ax = panels
    calls = []
    selector = ThresholdSelector(
        0.4, hist_ax=hist_ax, trace_ax=trace_ax,
        on_change=calls.append, on_release=calls.append,
    )

    selector.set_value(0.6, notify=False)

    assert selector.value == pytest.approx(0.6)
    assert line_positions(selector) == pytest.approx([0.6, 0.6])
    assert calls == []


def test_arrow_keys_nudge_and_r_resets(panels):
    _fig, hist_ax, trace_ax = panels
    releases = []
    selector = ThresholdSelector(
        0.4, hist_ax=hist_ax, trace_ax=trace_ax, step=0.01,
        on_release=releases.append,
    )

    press_key(hist_ax, "up")
    assert selector.value == pytest.approx(0.41)

    press_key(hist_ax, "shift+down")
    assert selector.value == pytest.approx(0.31)

    press_key(hist_ax, "r")
    assert selector.value == pytest.approx(0.4)
    assert releases == pytest.approx([0.41, 0.31, 0.4])


def test_one_view_is_enough(panels):
    _fig, hist_ax, _trace_ax = panels

    selector = ThresholdSelector(0.4, hist_ax=hist_ax)

    selector.set_value(0.6)
    assert line_positions(selector) == pytest.approx([0.6])


def test_no_view_at_all_is_refused(panels):
    with pytest.raises(ValueError, match="at least one axes"):
        ThresholdSelector(0.4)


def test_another_view_can_be_added_later(panels):
    _fig, hist_ax, trace_ax = panels
    selector = ThresholdSelector(0.4, hist_ax=hist_ax)

    selector.add_axis(trace_ax, "h")
    selector.set_value(0.7)

    # Nothing special-cases two views, so a third is free -- which is what
    # lets the no-signal threshold be a second instance rather than a branch.
    assert line_positions(selector) == pytest.approx([0.7, 0.7])


def test_disconnecting_stops_it_listening(panels):
    _fig, hist_ax, trace_ax = panels
    selector = ThresholdSelector(0.4, hist_ax=hist_ax, trace_ax=trace_ax)

    selector.disconnect()
    click(hist_ax, 0.4, 0.5)
    click(hist_ax, 0.8, 0.5, kind="motion_notify_event")

    assert selector.value == pytest.approx(0.4)


def test_it_knows_nothing_about_emg(panels):
    """The same class has to serve a Refinement threshold on a PC score."""
    _fig, hist_ax, trace_ax = panels
    hist_ax.set_xlim(-3, 3)
    trace_ax.set_ylim(-3, 3)

    selector = ThresholdSelector(0.0, hist_ax=hist_ax, trace_ax=trace_ax)
    selector.set_value(-1.5)

    assert selector.value == pytest.approx(-1.5)
