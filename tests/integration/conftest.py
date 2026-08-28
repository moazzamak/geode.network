"""Integration layer: multi-module in-process flows.

The repair-overlay test reads the corpus cache. On this machine the
global environment still points at a stale D: path; force the
known-good cache location here. On CI (non-Windows) the variable
stays unset, the cache root does not exist, and the cache-dependent
test skips itself (see docs/TESTING.md).
"""
import os
import sys

import pytest

pytestmark = pytest.mark.integration

if sys.platform == "win32":
    os.environ["GEODE_CACHE_DIR"] = r"F:\geode-ml\data\cache"
