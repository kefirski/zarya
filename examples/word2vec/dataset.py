import logging
import os
import random
import re
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
_re_tokenizer = re.compile(r"\W+")


class Word2VecDataset(Dataset):
    """Word2Vec word embedding dataset.
    This dataset must consist of raw text file.

    Args:
        corpus_file (str): corpus file name.
        vocab_size (int): vocabulary size.
        context_size (int): context size.
        negative_size (int): negative size.
    """

    def __init__(
        self, corpus_file: str, vocab_size: int, context_size: int, negative_size: int
    ):
        data_folder = os.path.dirname(corpus_file)
        data_name = os.path.splitext(os.path.basename(corpus_file))[0]
        dump_filename = os.path.join(data_folder, data_name + "_dump")

        # Check preprocessed data dump existence:
        if os.path.exists(dump_filename + ".npy"):
            logger.info(f"Loading preprocessed data dump `{dump_filename}`.")

            self.i2w, self.keep_probs, self.neg_probs, self.data = self.load(
                dump_filename + ".npy"
            )
            self.w2i = {word: i for i, word in enumerate(self.i2w)}
            self.context_pairs = np.array(
                [*self.build_context(self.data, context_size)]
            )
            self.negative = np.random.choice(
                len(self.i2w), (vocab_size, negative_size), p=self.neg_probs
            )

            logger.info(
                f"Loaded {len(self.context_pairs)} context pairs from dataset dump."
            )
        else:
            logger.info(f"Start processing `{corpus_file}` corpus.")

            word_counter = Counter(self.read_corpus_tokens(corpus_file))
            most_common_words = word_counter.most_common(vocab_size)

            self.keep_probs = self.subsampling_probs(most_common_words)
            self.neg_probs = self.negative_probs(most_common_words)

            self.i2w = [word for word, freq in most_common_words]
            self.w2i = {word: i for i, word in enumerate(self.i2w)}
            logger.info(f"Vocabulary of size {len(self.i2w)} has been builded.")

            data_index = [
                self.w2i[token]
                for token in self.read_corpus_tokens(corpus_file)
                if token in self.w2i
            ]
            # Apply subsampling
            self.data = [t for t in data_index if random.random() < self.keep_probs[t]]
            logger.info(f"Subsampling {len(self.data)} corpus words.")

            # Generate contexts and negatives:
            self.context_pairs = np.array(
                [*self.build_context(self.data, context_size)]
            )
            self.negative = np.random.choice(
                len(self.i2w), (vocab_size, negative_size), p=self.neg_probs
            )
            logger.info(f"Loaded {len(self.context_pairs)} context pairs.")

            self.dump(
                dump_filename, self.i2w, self.keep_probs, self.neg_probs, self.data
            )
            logger.info(f"Dataset has been dumped into `{dump_filename}.npy` file.")

    def __getitem__(self, index):
        center, context = self.context_pairs[index]
        negative = random.choice(self.negative)
        return center, context, negative

    def __len__(self):
        return len(self.context_pairs)

    @staticmethod
    def dump(dump_name, *args):
        data_dump = np.array(args)
        np.save(dump_name, data_dump)

    @staticmethod
    def load(dump_name):
        data = np.load(dump_name, allow_pickle=True)
        return tuple(data)

    @staticmethod
    def negative_probs(token_meta, alpha=0.75):
        words, count = zip(*token_meta)
        count = np.array(count) ** alpha
        unigram_distr = count / count.sum()
        return unigram_distr

    @staticmethod
    def subsampling_probs(token_meta, eps=1e-4):
        words, count = zip(*token_meta)
        count = np.array(count)
        freq = count / count.sum()
        prob = np.sqrt(eps / freq) + eps / freq
        return prob

    @staticmethod
    def build_context(data: list, window: int):
        for i, token in enumerate(data):
            lcontext = window // 2
            rcontext = window - lcontext
            lspan = slice(i - lcontext, i)
            rspan = slice(i + 1, i + rcontext + 1)
            for context in data[lspan] + data[rspan]:
                yield token, context

    @classmethod
    def read_corpus_tokens(cls, corpus_file: str):
        with open(corpus_file) as file:
            for line in file:
                yield from cls.tokenize(line.lower())

    @staticmethod
    def tokenize(string):
        return _re_tokenizer.split(string.lower())

    @staticmethod
    def get_collate():
        def collate(batch):
            cols = zip(*batch)
            tensors = tuple(map(torch.LongTensor, cols))
            return tensors

        return collate
