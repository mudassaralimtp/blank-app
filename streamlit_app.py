"""
streamlit_app.py
────────────────
Entry-point shim for Streamlit Community Cloud.

Streamlit Cloud looks for streamlit_app.py at the repo root by default.
This file simply imports and runs dashboard/app.py so all relative paths
and module imports inside app.py continue to work without modification.
"""

import runpy
from pathlib import Path

# Run dashboard/app.py as __main__ in its own namespace.
# Using runpy ensures sys.path manipulation inside app.py still fires first.
runpy.run_path(str(Path(__file__).parent / "dashboard" / "app.py"), run_name="__main__")
