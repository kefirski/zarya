import numpy as np

import torch
import argparse

from matplotlib import pyplot as plt
from random import sample
from zarya import manifolds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str)
    parser.add_argument("--vocab", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--manifold", choices=["poincare", "lorentz"], default="poincare"
    )
    args = parser.parse_args()

    plt.figure(figsize=(args.size, args.size))

    weights = np.load(args.weights)
    vocab = np.load(args.vocab)

    if args.manifold == "lorentz":
        mf = manifolds.LorentzManifold()
        weights = torch.from_numpy(weights)
        weights = mf.to_poincare(weights).numpy()

    assert len(weights.shape) == 2, "Weights must be a 2 dim array"

    s = sample(range(len(vocab)), min(args.limit, len(vocab)))
    for i in s:
        x, y = weights[i]
        title = vocab[i]
        plt.scatter(x, y, c="b")
        plt.annotate(title, (x, y))

    plt.savefig(args.out)
