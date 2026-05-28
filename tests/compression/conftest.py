import sys
from pathlib import Path

# pytest's pythonpath already adds research/, but `python -m pytest` from some
# CWDs may not — mirror the src/ belt-and-suspenders pattern in tests/conftest.py.
RESEARCH = Path(__file__).resolve().parents[2] / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))
