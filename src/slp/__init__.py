"""Standard Library of Primitives (SLP) -- public entry point.

This package is the developer-facing surface of the GEODE standard
library. A primitive is a deterministic, code-defined transform
that implements the :class:`~src.primitive.Primitive` interface.

Findability rules (registered 25 Aug):

- The living index is ``src/slp/CATALOG.md``. It is generated from
  :func:`catalog` and checked in. Browse it to find primitives.
- The developer guide is ``src/slp/README.md``. It states the
  naming conventions, the determinism contract, and the checklist
  for adding a primitive.
- Every built primitive has a ``PrimitiveSpec`` (name, category,
  params, description) and a factory docstring in STE style.
- Import primitives from this package. Do not import the
  underlying modules directly.

PENDING entries mark the launch backlog. Their code ships when
they are built. Until then the catalog says so explicitly. The
full possibility space lives in
``analysis/SLP_POSSIBILITY_SPACE_v1.md``.
"""

from geode.core.audio_primitives import (
    SAMPLE_RATE as MEL_SAMPLE_RATE,
    mel_spectrogram,
    primitive_replay_hash,
)
from src.primitive import (
    Primitive,
    PrimitiveSpec,
    make_affine,
    make_clip,
    make_delay,
    make_l2_normalize,
    make_logical_and,
    make_logical_or,
    make_scale,
    make_select_dims,
    make_threshold,
)
from src.programmatic_memory import Continuation, ProgrammaticMemory
from src.programmatic_primitive import PrimitiveContract, ProgrammaticPrimitive

from src.slp._catalog import STATUS_BUILT, STATUS_PENDING, catalog

__all__ = [
    # classes
    "Primitive",
    "PrimitiveSpec",
    "PrimitiveContract",
    "ProgrammaticPrimitive",
    "ProgrammaticMemory",
    "Continuation",
    # built math primitives
    "make_scale",
    "make_clip",
    "make_affine",
    "make_l2_normalize",
    # built logic primitives
    "make_threshold",
    "make_logical_and",
    "make_logical_or",
    # built transform primitives
    "make_select_dims",
    # built signal primitives
    "make_delay",
    "mel_spectrogram",
    "MEL_SAMPLE_RATE",
    "primitive_replay_hash",
    # catalog
    "catalog",
    "STATUS_BUILT",
    "STATUS_PENDING",
]
