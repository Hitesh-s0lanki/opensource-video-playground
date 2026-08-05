"""Streamlit entry point.

    streamlit run app.py

Kept at the project root so the run command stays short; the real UI lives in
``src/vidgen/app/streamlit_app.py``.
"""

from pathlib import Path
from runpy import run_path

_APP = Path(__file__).resolve().parent / "src" / "vidgen" / "app" / "streamlit_app.py"

run_path(str(_APP), run_name="__main__")
