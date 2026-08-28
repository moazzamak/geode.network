"""Local compatibility patch 2 for speechbrain on Windows without
symlink privilege: the default local fetch strategy becomes COPY
instead of SYMLINK (registered 24 Aug, the M288 unblock)."""
import pathlib

F = pathlib.Path(r"C:\Users\mak\Documents\projects\Research Work"
                r"\CG-MoE\.venv-rocm\Lib\site-packages\speechbrain"
                r"\utils\fetching.py")
T = F.read_text(encoding="utf-8")
OLD = "    local_strategy: LocalStrategy = LocalStrategy.SYMLINK,\n"
NEW = "    local_strategy: LocalStrategy = LocalStrategy.COPY,\n"
if OLD not in T:
    raise SystemExit("fetching signature not found — inspect fetching.py")
F.write_text(T.replace(OLD, NEW), encoding="utf-8")
print("patched fetching default strategy to COPY")
