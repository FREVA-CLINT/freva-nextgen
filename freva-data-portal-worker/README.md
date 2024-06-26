# Data Loader

The Data Loader is a Python package designed to open, chunk, and cache
climate datasets from various storage locations.
By using the power of dask-distributed, this library ensures efficient data
processing and accessibility. Cached data is stored in Redis and can be
accessed via a REST API for Zarr endpoint streaming.

## Features

- Open and process climate datasets from multiple storage locations
- Chunk data for optimized performance
- Cache processed data in Redis
- Access cached data through a REST API for Zarr endpoint streaming

## Installation

To install the Data Loader package, use the following command:

```console
python3 -m pip install -e .
```
