from string import ascii_lowercase

import numpy as np
import torch as t


class Dataloader:
    def __init__(self, data_path=""):
        """
        :param data_path: path to data
        """

        assert isinstance(
            data_path, str
        ), "Invalid data_path type. Required {}, but {} found".format(
            str, type(data_path)
        )

        with open("{}text8".format(data_path)) as file:
            self.data = file.read()

        self.data = {
            "train": self.data[: int(90e6)],
            "valid": self.data[int(90e6) : int(95e6)],
            "test": self.data[int(95e6) :],
        }

        self.idx_to_char = [" "] + [c for c in ascii_lowercase]
        self.char_to_idx = {c: i for i, c in enumerate(self.idx_to_char)}

    def next_batch(self, batch_size, seq_len, target, device):
        data = self.data[target]

        indices = np.random.randint(len(data) - seq_len - 2, size=batch_size)

        input = [data[idx : idx + seq_len + 1] for idx in indices]
        input = [[self.char_to_idx[c] for c in batch] for batch in input]

        target = [batch[1:] for batch in input]
        input = [batch[:-1] for batch in input]

        return tuple(
            [t.tensor(i, dtype=t.long, device=device) for i in [input, target]]
        )
