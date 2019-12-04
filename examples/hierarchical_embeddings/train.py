import argparse
import logging
import multiprocessing
from datetime import datetime

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HierarchicalDataset
from model import EntailmentConesEmbeddings, HierarchicalEmbeddings
from utils import setup_logging
from zarya.manifolds import LorentzManifold, PoincareBall
from zarya.optim import RSGD

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--out", type=str)
    parser.add_argument("--vocab", type=int, default=30000)
    parser.add_argument("--emb", type=int, default=2)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--negatives", type=int, default=10)
    parser.add_argument("--closure", action="store_true")
    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--cones", action="store_true")
    parser.add_argument(
        "--manifold", choices=["poincare", "lorentz"], default="poincare"
    )
    parser.add_argument("--retraction", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    torch.set_num_threads(args.threads)

    dataset = HierarchicalDataset(
        args.source, args.vocab, args.negatives, closure=args.closure
    )
    dataloader = DataLoader(
        dataset, args.bs, True, collate_fn=dataset.collate, num_workers=args.threads
    )
    pbar = tqdm(total=len(dataloader) * args.epochs, desc="Trainig")

    manifold = PoincareBall() if args.manifold == "poincare" else LorentzManifold()
    model_cls = EntailmentConesEmbeddings if args.cones else HierarchicalEmbeddings
    model = model_cls(dataset.vocab_size, args.emb, manifold).to(device)
    optim = RSGD(model.parameters(), manifold, args.lr, args.retraction)
    # Burn Up scheduler
    lr_scheduler = LambdaLR(optim, lambda epoch: 0.1 if epoch < 10 else 1)

    try:
        for epoch in range(1, args.epochs + 1):
            cum_loss = 0
            for batch in dataloader:
                pred, succ, neg = [t.to(device) for t in batch]

                optim.zero_grad()
                loss = model(pred, succ, neg)
                loss.backward()
                optim.step()
                model.renorm()

                cum_loss += loss.item()
                pbar.update()

            lr_scheduler.step()
            logger.info(f"Epoch {epoch} loss: {cum_loss / len(dataloader)}")
    except KeyboardInterrupt:
        logger.warning("Force stopping")

    vocab = np.array(dataset.i2w)
    weights = np.array(model.dump())
    time = datetime.now().strftime("%y%m%d_%H%M%S")
    np.save(f"{args.out}_vocab", vocab)
    np.save(f"{args.out}_{time}", weights)
