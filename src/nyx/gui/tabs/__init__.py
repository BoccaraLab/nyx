"""One tab per decision the notebooks stop at.

The order here is the order in the side rail, and it is the order of
``examples/03_score_rodents.ipynb`` -- with the notebook's PCA, clustering and
naming STOPs merged into :class:`~nyx.gui.tabs.sleep.SleepTab`, because in
practice they are one loop rather than three steps.
"""

from __future__ import annotations

from nyx.gui.tabs.base import Tab
from nyx.gui.tabs.emg import EmgTab
from nyx.gui.tabs.load import LoadTab
from nyx.gui.tabs.result import ResultTab
from nyx.gui.tabs.signal import SignalTab
from nyx.gui.tabs.sleep import SleepTab

__all__ = ["Tab", "LoadTab", "SignalTab", "EmgTab", "SleepTab", "ResultTab", "TABS"]

#: In rail order.
TABS = [LoadTab, SignalTab, EmgTab, SleepTab, ResultTab]
