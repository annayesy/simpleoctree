import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if sys.path[0] != root_str:
    try:
        sys.path.remove(root_str)
    except ValueError:
        pass
    sys.path.insert(0, root_str)
