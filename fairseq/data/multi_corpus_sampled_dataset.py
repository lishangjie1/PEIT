# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
from typing import Callable, Dict, List

import numpy as np

from . import FairseqDataset


def uniform_sampler(x):
    # Sample from uniform distribution
    return np.random.choice(x, 1).item()


class MultiCorpusSampledDataset(FairseqDataset):
    """
    Stores multiple instances of FairseqDataset together and in every iteration
    creates a batch by first sampling a dataset according to a specified
    probability distribution and then getting instances from that dataset.

    Args:
        datasets: an OrderedDict of FairseqDataset instances.
        sampling_func: A function for sampling over list of dataset keys.
            The default strategy is to sample uniformly.
    """

    def __init__(
        self,
        datasets: Dict[str, FairseqDataset],
        sampling_func: Callable[[List], int] = None,
    ):
        super().__init__()
        assert isinstance(datasets, OrderedDict)
        self.datasets = datasets
        if sampling_func is None:
            sampling_func = uniform_sampler
        self.sampling_func = sampling_func

        self.total_num_instances = 0
        for _, dataset in datasets.items():
            assert isinstance(dataset, FairseqDataset)
            self.total_num_instances += len(dataset)

        self._ordered_indices = None

    def __len__(self):
        """
        Length of this dataset is the sum of individual datasets
        """
        return self.total_num_instances

    def ordered_indices(self):
        """
        Ordered indices for batching. Here we call the underlying
        dataset's ordered_indices() so that we get the same random ordering
        as we would have from using the underlying dataset directly.
        """
        if self._ordered_indices is None:
            self._ordered_indices = OrderedDict(
                [
                    (key, dataset.ordered_indices())
                    for key, dataset in self.datasets.items()
                ]
            )
        return np.arange(len(self))

    def _map_index_to_dataset(self, key: int, index: int):
        """
        Different underlying datasets have different lengths. In order to ensure
        we are not accessing an index outside the range of the current dataset
        size, we wrap around. This function should be called after we have
        created an ordering for this and all underlying datasets.
        """
        assert (
            self._ordered_indices is not None
        ), "Must call MultiCorpusSampledDataset.ordered_indices() first"
        mapped_index = index % len(self.datasets[key])
        return self._ordered_indices[key][mapped_index]

    def __getitem__(self, index: int):
        """
        Get the item associated with index from each underlying dataset.
        Since index is in the range of [0, TotalNumInstances], we need to
        map the index to the dataset before retrieving the item.
        """
        return OrderedDict(
            [
                (key, dataset[self._map_index_to_dataset(key, index)])
                for key, dataset in self.datasets.items()
            ]
        )

    def collater(self, samples: List[Dict]):
        """
        Generate a mini-batch for this dataset.
        To convert this into a regular mini-batch we use the following
        logic:
            1. Select a dataset using the specified probability distribution.
            2. Call the collater function of the selected dataset.
        """
        if len(samples) == 0:
            return None

        selected_key = self.sampling_func(list(self.datasets.keys()))
        selected_samples = [sample[selected_key] for sample in samples]
        return self.datasets[selected_key].collater(selected_samples)

    def num_tokens(self, index: int):
        """
        Return an example's length (number of tokens), used for batching. Here
        we return the max across all examples at index across all underlying
        datasets.
        """
        return max(
            dataset.num_tokens(self._map_index_to_dataset(key, index))
            for key, dataset in self.datasets.items()
        )
    
    def num_tokens_vec(self, indices):
        num_tokens_per_dataset = []
        for key in self.datasets:
            dataset = self.datasets[key]
            map_indices = self._ordered_indices[key][indices % len(dataset)]
            num_tokens_per_dataset.append(dataset.num_tokens_vec(map_indices))
        
        return np.stack(num_tokens_per_dataset).max(axis=0)
            


    def size(self, index: int):
        """
        Return an example's size as a float or tuple. Here we return the max
        across all underlying datasets. This value is used when filtering a
        dataset with max-positions.
        """
        # return max(
        #     dataset.size(self._map_index_to_dataset(key, index))
        #     for key, dataset in self.datasets.items()
        # )
        sizes = (
            dataset.size(self._map_index_to_dataset(key, index))
            for key, dataset in self.datasets.items()
        )

        max_src_size, max_tgt_size = 0, 0
        for size in sizes:
            max_src_size = max(max_src_size, size[0])
            max_tgt_size = max(max_tgt_size, size[1])
        
        return (max_src_size, max_tgt_size)

    def filter_indices_by_size(self, indices, max_sizes):
        from fairseq.data import data_utils
        # merge all invalid indices
        ignored = set([])
        for key in self.datasets:
            dataset = self.datasets[key]
            map_indices = self._ordered_indices[key]

            single_indices, single_ignored = data_utils.filter_paired_dataset_indices_by_size(
                dataset.src_sizes[map_indices],
                dataset.tgt_sizes[map_indices],
                np.arange(len(dataset)),
                max_sizes,
            )
            for ignore in single_ignored:
                while ignore < len(self):
                    ignored.add(ignore)
                    ignore += len(dataset)
            
        

        indices = np.array([i for i in indices if i not in ignored])
        ignored = list(ignored)
        return indices, ignored



    @property
    def supports_prefetch(self):
        return all(
            getattr(dataset, "supports_prefetch", False)
            for dataset in self.datasets.values()
        )

    def prefetch(self, indices):
        for key, dataset in self.datasets.items():
            dataset.prefetch(
                [self._map_index_to_dataset(key, index) for index in indices]
            )

    @property
    def supports_fetch_outside_dataloader(self):
        return all(
            self.datasets[key].supports_fetch_outside_dataloader
            for key in self.datasets
        )
