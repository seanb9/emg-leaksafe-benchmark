"""ReAble EMG pipeline v2.

A clean, leak-free, reproducible re-implementation of the cross-subject EMG
gesture-recognition pipeline. See ml-v2/PLAN.md for the design rationale and
ml-v2/CURRENT_STATE.md for the v1 baseline this replaces.
"""

__version__ = "2.0.0-dev"

# Force UTF-8 console output so logs containing non-ASCII glyphs (λ, ±, →, —) don't
# crash on Windows, whose default console encoding (cp1252) can't encode them.
# Every CLI imports emgv2, so doing it here covers all entry points.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass
