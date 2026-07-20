"""Setup smoke test: the package imports and exposes a non-empty version.

Keeps `pytest` green from day one so every later slice has a baseline to build on.
"""

from __future__ import annotations

import mlfactory


def test_version_present():
    assert isinstance(mlfactory.__version__, str)
    assert mlfactory.__version__
