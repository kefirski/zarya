import numpy as np

import argparse

from matplotlib import pyplot as plt
from random import sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str)
    parser.add_argument("--vocab", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()

    plt.figure(figsize=(args.size, args.size))

    weights = np.load(args.weights)
    vocab = np.load(args.vocab)
    s = sample(range(len(vocab)), min(args.limit, len(vocab)))
    for i in s:
        x, y = weights[i]
        title = vocab[i]
        plt.scatter(x, y, c="b")
        plt.annotate(title, (x, y))

    plt.savefig(args.out)
