"""Pin ROCm to the discrete GPU before PyTorch initializes.

The Ryzen 7800X3D exposes an integrated ``gfx1036`` adapter that ROCm
enumerates as device 0, ahead of the discrete RX 9070 XT (``gfx1201``). The
installed ``gfx120X-all`` wheel ships no kernels for ``gfx1036``, and leaving
the integrated adapter visible is not merely a matter of choosing a different
device index: HIP loads code objects for the whole visible set, so device-side
kernels fail with ``hipErrorInvalidKernelFile`` even when the discrete card is
selected explicitly.

The only reliable fix is to mask the integrated adapter out of the process
before the HIP runtime starts. Import this module and call
:func:`ensure_discrete_gpu` at the very top of a script, ahead of ``import
torch``.
"""

from __future__ import annotations

import os
import subprocess
import sys

#: Architectures the installed ROCm wheel ships kernels for.
SUPPORTED_ARCHITECTURE_PREFIX = "gfx120"

#: Set on the re-executed child so it never recurses.
_GUARD_VARIABLE = "GEODE_ROCM_DEVICE_PINNED"

_ENUMERATION_SOURCE = """
import torch
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_properties(index).gcnArchName)
"""


def discrete_gpu_index(*, architecture_prefix: str = SUPPORTED_ARCHITECTURE_PREFIX) -> int | None:
    """Return the index of the first supported GPU, or ``None`` if there is none.

    Enumeration runs in a child process so that the calling process does not
    initialize HIP with the integrated adapter visible.
    """
    environment = dict(os.environ)
    environment.pop("HIP_VISIBLE_DEVICES", None)
    environment.pop("ROCR_VISIBLE_DEVICES", None)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _ENUMERATION_SOURCE],
            capture_output=True,
            text=True,
            env=environment,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in completed.stdout.splitlines():
        index, _, architecture = line.partition(" ")
        if architecture.strip().startswith(architecture_prefix):
            return int(index)
    return None


def ensure_discrete_gpu(*, architecture_prefix: str = SUPPORTED_ARCHITECTURE_PREFIX) -> None:
    """Mask out unsupported adapters, re-executing the process if necessary.

    Does nothing when the caller has already pinned a device, when this
    process is the re-executed child, or when no supported GPU is present --
    in the last case the caller is expected to fall back to the CPU.
    """
    if os.environ.get(_GUARD_VARIABLE) or "HIP_VISIBLE_DEVICES" in os.environ:
        return
    if "torch" in sys.modules:
        raise RuntimeError(
            "ensure_discrete_gpu() must be called before 'torch' is imported; "
            "the HIP runtime has already enumerated its devices."
        )

    index = discrete_gpu_index(architecture_prefix=architecture_prefix)
    if index is None:
        os.environ[_GUARD_VARIABLE] = "no-supported-gpu"
        return

    environment = dict(os.environ)
    environment["HIP_VISIBLE_DEVICES"] = str(index)
    environment[_GUARD_VARIABLE] = str(index)
    completed = subprocess.run(  # noqa: S603 - re-executing this interpreter
        [sys.executable, *sys.argv], env=environment, check=False
    )
    raise SystemExit(completed.returncode)
