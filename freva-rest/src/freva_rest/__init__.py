import os
from pathlib import Path

__version__ = "2412.0.0-dev2"
__all__ = ["__version__"]

REST_URL = (
    os.environ.get("API_URL")
    or f"http://localhost:{os.environ.get('API_PORT', '8080')}"
)
CACHE_EXP = os.environ.get("API_CACHE_EXP") or "3600"
TMP_DIR = Path(os.environ.get("API_TMP_DIR") or "/tmp")
DASK_CLUSTER = os.environ.get("API_CLUSTER", "LocalCluster")

if __name__ == "__main__":
    print(__version__)
