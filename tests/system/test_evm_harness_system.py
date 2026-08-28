"""System test: the EVM harness suite (Hardhat) passes in full.

This is the cross-process acceptance gate for the contracts: 70 tests
covering GeodeToken, VestingVault, CreditLedger, LinearProofVerifier,
and ProofAnchor. Skipped when Node/Hardhat is unavailable.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVM_DIR = REPO_ROOT / "infrastructure" / "evm"

pytestmark = pytest.mark.system


@pytest.fixture(scope="module")
def node_available() -> bool:
    return shutil.which("npx") is not None


def test_evm_harness_suite(node_available: bool) -> None:
    if not node_available:
        pytest.skip("npx is not available on this machine")
    proc = subprocess.run(
        ["npx", "hardhat", "test"], cwd=EVM_DIR, capture_output=True,
        text=True, shell=True, timeout=600,
    )
    assert proc.returncode == 0, (
        "the EVM harness suite failed:\n"
        f"{(proc.stdout or '')[-1500:]}\n{(proc.stderr or '')[-1500:]}")
