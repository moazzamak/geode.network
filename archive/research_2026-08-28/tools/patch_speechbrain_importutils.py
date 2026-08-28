"""Local compatibility patch for speechbrain on the ROCm torch 2.11
Windows build (registered 24 Aug, the M288 unblock): the lazy
import of optional integrations (k2_fsa, huggingface.wordemb) is
unavailable here; ensure_module returns a stub instead of raising
so the xvector encoder path loads. Attribute access on the stub
fails only if the integration is actually used."""
import pathlib

F = pathlib.Path(r"C:\Users\mak\Documents\projects\Research Work"
                r"\CG-MoE\.venv-rocm\Lib\site-packages\speechbrain"
                r"\utils\importutils.py")
T = F.read_text(encoding="utf-8")
OLD = ('            except Exception as e:\n'
       '                raise ImportError(f"Lazy import of '
       '{repr(self)} failed") from e\n')
NEW = ('            except Exception as e:\n'
       '                # local compatibility (torch 2.11 ROCm '
       'Windows): optional\n'
       '                # integrations are unavailable; a stub is '
       'returned and\n'
       '                # attribute access fails only if actually used\n'
       '                import types as _types\n'
       '                self.lazy_module = _types.ModuleType(self.target)\n'
       '                self.lazy_module.__doc__ = (\n'
       '                    "unavailable optional integration: "\n'
       '                    + repr(e))\n')
if OLD not in T:
    raise SystemExit("target block not found — inspect importutils.py")
F.write_text(T.replace(OLD, NEW), encoding="utf-8")
print("patched importutils.ensure_module")
