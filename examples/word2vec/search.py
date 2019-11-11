import argparse

import numpy as np
import torch
from torch.nn import functional


def most_common(word_id: int, n: int, weights: torch.Tensor):
    n_vec, vec_size = weights.shape
    word_vector = weights[word_id].repeat(n_vec, 1)
    scores = functional.cosine_similarity(word_vector, weights)
    return scores.topk(n, largest=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("words", nargs="+", type=str)
    parser.add_argument("--weights", type=str, default="data/text8/weights.npy")
    parser.add_argument("--vocab", type=str, default="data/text8/weights-vocab.npy")
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    i2w = np.load(args.vocab)
    weights = np.load(args.weights)
    weights = torch.from_numpy(weights)
    w2i = {word: i for i, word in enumerate(i2w)}

    for word in args.words:
        print(f"--- {word} ---")
        word_id = w2i[word]
        scores, idxs = most_common(word_id, args.n, weights)
        for i, index in enumerate(idxs):
            print(f"{scores[i]:.4f} : {i2w[index]}")
