import csv
import logging
import random

import numpy as np
import torch
from scipy.sparse import csgraph, csr_matrix
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class HierarchicalDataset(Dataset):
    """Hierarchical Dataset.

    Args:
        source_file (str): Source filename with hierarchy pairs, separated
            by comma.
        negatives (int, optional): Number of negative samples. Defaults to 10.
        negative_bucket (int, optional): Temporary negative sample dump size.
            Defaults to 20000.
        closure (bool, optional): Whether to build transitive closure for
            hierarchy. Defaults to False.
    """

    def __init__(self, source_file, negatives=10, negative_bucket=20000, closure=False):
        logger.info(f"Start processing `{source_file}` hierarchy.")

        unique_tokens = set(t for p in self.iter_data(source_file) for t in p)
        self.i2w = list(unique_tokens)
        self.w2i = {w: i for i, w in enumerate(self.i2w)}

        self.data = np.array(
            [(self.w2i[p], self.w2i[s]) for p, s in self.iter_data(source_file)]
        )

        if closure:
            _last_len = len(self.data)
            self.data = self.transitive_closure(self.data, len(self.w2i))
            logger.info(f"Applied transitive closure: {_last_len} -> {len(self.data)}")

        self.negative = np.random.choice(len(self.w2i), (negative_bucket, negatives))

        logger.info(f"Loaded {len(self)} hierarchical pairs.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pred, succ = self.data[index]
        negative = random.choice(self.negative)
        return pred, succ, negative

    @property
    def vocab_size(self):
        return len(self.w2i)

    def export_vocab(self, filename):
        vocab = np.array(self.i2w)
        np.save(filename, vocab)

    @classmethod
    def iter_data(cls, source_file: str):
        with open(source_file) as file:
            reader = csv.reader(file)
            yield from reader

    @staticmethod
    def transitive_closure(data, size):
        M = csr_matrix(([True] * len(data), zip(*data)), shape=(size, size))
        paths = csgraph.shortest_path(M, directed=True)
        paths[~np.isfinite(paths)] = 0
        nonzero = paths.nonzero()
        closure = [*zip(*nonzero)]
        return closure

    @staticmethod
    def collate(batch):
        cols = zip(*batch)
        tensors = tuple(map(torch.LongTensor, cols))
        return tensors
