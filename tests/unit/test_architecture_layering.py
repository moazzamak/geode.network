"""M216 — executable architecture rules (fitness functions).

The single source of truth for these rules is the dependency table in
``docs/ARCHITECTURE.md``:

    hashing        no geode imports
    audit          -> hashing
    core           -> audit, hashing, core
    attribution    -> audit, hashing, core, attribution
    settlement     -> audit, hashing, core, settlement
    privacy        -> audit, hashing, core, privacy
    api            -> any geode layer (the application layer)

plus three global rules:

    R1  geode never imports ``experiments.*`` (experiments -> geode
        only);
    R2  no module uses a flat pre-M215 path (``geode.<module>``
        without a subpackage);
    R3  the public API surface (``geode.__all__``) resolves entirely.

These rules are asserted from the module graph itself, so the
architecture stays enforced as the codebase grows.
"""
from __future__ import annotations

import ast
import importlib
import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
GEODE = REPO / "geode"

LAYERS = ("hashing", "audit", "core", "attribution", "settlement",
          "privacy", "api")

# what each layer may import, by subpackage name (the direction table)
ALLOWED = {
    "hashing": set(),
    "audit": {"hashing"},
    "core": {"audit", "hashing", "core"},
    "attribution": {"audit", "hashing", "core", "attribution"},
    "settlement": {"audit", "hashing", "core", "settlement"},
    "privacy": {"audit", "hashing", "core", "privacy"},
    "api": {"audit", "hashing", "core", "attribution", "settlement",
            "privacy", "api"},  # the application layer
}


def _module_files() -> list[tuple[str, pathlib.Path]]:
    """(layer, path) for every product module (package __init__ files
    are included; the direction test skips them because a package init
    aggregates its own layer by design)."""
    out: list[tuple[str, pathlib.Path]] = [("hashing", GEODE / "hashing.py")]
    for sub in LAYERS[1:]:
        for p in sorted((GEODE / sub).rglob("*.py")):
            out.append((sub, p))
    return out


def _geode_imports(path: pathlib.Path) -> list[tuple[str, str]]:
    """[(module, lineno), ...] for every geode import in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.startswith("geode."):
            found.append((node.module, node.lineno))
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("geode."):
                    found.append((alias.name, node.lineno))
    return found


def _has_experiments_import(path: pathlib.Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
            and node.module.startswith("experiments")]


def _layer_of_import(target: str) -> str:
    return target.split(".")[1] if target.count(".") >= 1 else target


class TestNoExperimentsImports(unittest.TestCase):
    def test_geode_never_imports_experiments(self) -> None:
        violations = []
        for layer, path in _module_files():
            for lineno in _has_experiments_import(path):
                violations.append(f"{path}:{lineno}")
        self.assertEqual(violations, [],
                         f"geode imports experiments.* at: {violations}")


class TestNoFlatModulePaths(unittest.TestCase):
    def test_every_import_is_subpackaged(self) -> None:
        violations = []
        for layer, path in _module_files():
            for target, lineno in _geode_imports(path):
                if _layer_of_import(target) not in LAYERS:
                    violations.append(f"{path}:{lineno} imports "
                                      f"flat path {target!r}")
        self.assertEqual(violations, [],
                         f"flat pre-M215 paths remain: {violations}")


class TestLayerDirection(unittest.TestCase):
    def test_imports_respect_the_direction_table(self) -> None:
        violations = []
        for layer, path in _module_files():
            for target, lineno in _geode_imports(path):
                if path.name == "__init__.py":
                    continue  # package init aggregates its own layer
                used = _layer_of_import(target)
                if used not in ALLOWED[layer]:
                    violations.append(
                        f"{path}:{lineno} layer {layer!r} imports "
                        f"{used!r} (allowed: {sorted(ALLOWED[layer])})")
        self.assertEqual(violations, [], f"direction violations: "
                         f"{violations}")

    def test_every_import_path_resolves(self) -> None:
        unresolved = []
        for layer, path in _module_files():
            for target, lineno in _geode_imports(path):
                try:
                    importlib.import_module(target)
                except ModuleNotFoundError:
                    unresolved.append(f"{path}:{lineno} -> {target}")
        self.assertEqual(unresolved, [],
                         f"unresolvable geode imports: {unresolved}")


class TestPublicApi(unittest.TestCase):
    def test_all_names_resolve(self) -> None:
        import geode
        missing = [name for name in geode.__all__
                   if not hasattr(geode, name)]
        self.assertEqual(missing, [],
                         f"geode.__all__ names missing: {missing}")

    def test_version_is_semver(self) -> None:
        import geode
        parts = geode.__version__.split(".")
        self.assertEqual(len(parts), 3, geode.__version__)
        self.assertTrue(all(p.isdigit() for p in parts), geode.__version__)


if __name__ == "__main__":
    unittest.main()
