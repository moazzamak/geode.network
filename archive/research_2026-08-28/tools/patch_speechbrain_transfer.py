"""Local compatibility patch 3 for speechbrain on Windows without
symlink privilege: the Pretrainer's collect_files default strategy
becomes COPY instead of SYMLINK (registered 24 Aug, M288)."""
import pathlib

F = pathlib.Path(r"C:\Users\mak\Documents\projects\Research Work"
                r"\CG-MoE\.venv-rocm\Lib\site-packages\speechbrain"
                r"\utils\parameter_transfer.py")
T = F.read_text(encoding="utf-8")
OLD = "        local_strategy=LocalStrategy.SYMLINK,\n"
NEW = "        local_strategy=LocalStrategy.COPY,\n"
if OLD not in T:
    raise SystemExit("collect_files signature not found")
F.write_text(T.replace(OLD, NEW), encoding="utf-8")
print("patched collect_files default strategy to COPY")
