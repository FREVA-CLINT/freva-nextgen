"""Aggregate datasets."""

import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, FrozenSet, List, Tuple, Union

import xarray as xr

from data_portal_worker.backends import load_data
from data_portal_worker.utils import data_logger


class XarrayAggregator:
    """
    Initialize the aggregator with a list of dataset file paths or xarray.Dataset objects.

    Parameters
    ----------
    datasets (List[Union[str, xr.Dataset]]):
        List of NetCDF file paths or xarray.Dataset objects.
    max_workers (int):
        Number of threads for parallel dataset loading.
    """

    def __init__(self, datasets: List[Union[str, xr.Dataset]]):

        self.raw_datasets = datasets
        self.datasets: List[xr.Dataset] = []
        self.aggregated_datasets: List[xr.Dataset] = []
        self.max_workers = self._get_num_threads(len(datasets))

    @staticmethod
    def _get_num_threads(num_datasets: int) -> int:
        """Calculate the max num of threads."""
        num_threads = (os.cpu_count() or 4) * 2 - 1
        return max(1, min(num_threads, num_datasets))

    def _load_dataset(
        self, dataset: Union[str, xr.Dataset]
    ) -> Union[xr.Dataset, None]:
        """
        Load a dataset from file or return the dataset if it's already an xarray object.

        Args:
            dataset (Union[str, xr.Dataset]): File path or xarray dataset.

        Returns:
            xr.Dataset or None: The loaded dataset, or None if loading fails.
        """
        if isinstance(dataset, xr.Dataset):
            return dataset

        try:
            return load_data(dataset)
        except Exception as e:
            data_logger.error(f"Failed to load dataset {dataset}: {e}")
            return None

    def load_datasets(self):
        """
        Load all datasets in parallel using ThreadPoolExecutor.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_dataset = {
                executor.submit(self._load_dataset, ds): ds
                for ds in self.raw_datasets
            }

            for future in as_completed(future_to_dataset):
                ds = future.result()
                if ds:
                    self.datasets.append(ds)

        data_logger.debug(
            f"Successfully loaded {len(self.datasets)} datasets out of {len(self.raw_datasets)}"
        )

    def group_datasets(
        self,
    ) -> Dict[
        Tuple[FrozenSet[Tuple[str, int]], FrozenSet[str]], List[xr.Dataset]
    ]:
        """
        Group datasets based on their dimensions and coordinate structure.

        Returns:
            Dict[Tuple[FrozenSet, FrozenSet], List[xr.Dataset]]: Grouped datasets.
        """
        grouped = defaultdict(list)
        for ds in self.datasets:
            key = (frozenset(ds.dims.items()), frozenset(ds.coords.keys()))
            grouped[key].append(ds)

        return grouped

    def merge_datasets(
        self,
        grouped_datasets: Dict[Tuple[FrozenSet, FrozenSet], List[xr.Dataset]],
    ) -> List[xr.Dataset]:
        """
        Attempt to merge datasets within each group.

        Args:
            grouped_datasets (Dict[Tuple[FrozenSet, FrozenSet], List[xr.Dataset]]): Grouped datasets.

        Returns:
            List[xr.Dataset]: Merged datasets.
        """
        merged_datasets = []
        for key, group in grouped_datasets.items():
            try:
                merged = xr.merge(
                    group
                )  # Merge different variables in the same grid
                merged_datasets.append(merged)
                data_logger.debug(
                    f"Merged {len(group)} datasets into one dataset"
                )
            except Exception as e:
                data_logger.warning(f"Failed to merge some datasets: {e}")
                merged_datasets.extend(group)  # Keep unmerged datasets separate

        return merged_datasets

    def concat_datasets(self, datasets: List[xr.Dataset]) -> List[xr.Dataset]:
        """
        Try to concatenate datasets along a shared dimension (e.g., 'time').

        Args:
            datasets (List[xr.Dataset]): List of merged datasets.

        Returns:
            List[xr.Dataset]: Final concatenated datasets.
        """
        final_datasets = []
        while datasets:
            ds = datasets.pop(0)
            same_grid = [d for d in datasets if set(ds.coords) == set(d.coords)]

            for d in same_grid:
                datasets.remove(d)
                try:
                    ds = xr.concat(
                        [ds, d], dim="time"
                    )  # Concatenation on time if possible
                    data_logger.debug(
                        f"Concatenated datasets along time dimension"
                    )
                except Exception as e:
                    data_logger.warning(f"Failed to concatenate datasets: {e}")
                    final_datasets.append(d)

            final_datasets.append(ds)

        return final_datasets

    def aggregate(self) -> List[xr.Dataset]:
        """
        Perform dataset aggregation by loading, grouping, merging, and concatenating datasets.

        Returns:
            List[xr.Dataset]: The minimal number of aggregated datasets.
        """
        data_logger.debug("Starting dataset aggregation process...")

        # Load datasets in parallel
        self.load_datasets()

        # Group datasets by dimensions and coordinates
        grouped_datasets = self.group_datasets()

        # Merge datasets within each group
        merged_datasets = self.merge_datasets(grouped_datasets)

        # Concatenate datasets along time (or other shared dimensions)
        self.aggregated_datasets = self.concat_datasets(merged_datasets)

        data_logger.debug(
            f"Aggregation completed: {len(self.aggregated_datasets)} final datasets created."
        )
        return self.aggregated_datasets


if __name__ == "__main__":
    import requests

    res = requests.get(
        "http://localhost:7777/api/freva-nextgen/databrowser/data-search/freva/file/",
        params={"project": "nukleus"},
        stream=True,
    )
    files = [r.decode("utf-8") for r in res.iter_lines()]
    aggregator = XarrayAggregator(files)
    final_datasets = aggregator.aggregate()
    for i, ds in enumerate(final_datasets):
        print(i, ds)
