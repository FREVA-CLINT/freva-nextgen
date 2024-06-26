"""Client software freva evaluation system framework (freva):

Freva, the free evaluation system framework, is a data search and analysis
platform developed by the atmospheric science community for the atmospheric
science community. With help of Freva researchers can:

- quickly and intuitively search for data stored at typical data centers that
  host many datasets.
- create a common interface for user defined data analysis tools.
- apply data analysis tools in a reproducible manner.

The code described here is currently in testing phase. The client and server
library described in the documentation only support searching for data. If you
need to apply data analysis plugins, please visit the
"""

from .auth import authenticate
from .query import databrowser

__version__ = "2404.0.1"
__all__ = ["authenticate", "databrowser", "__version__"]
