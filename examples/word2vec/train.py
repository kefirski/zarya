import argparse
import logging

import numpy as np
import torch
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Word2VecDataset
from model import Word2Vec
from utils import setup_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/text8")
    parser.add_argument("--output", type=str, default="data/weights")
    parser.add_argument("--emb", type=int, default=100)
    parser.add_argument("--vocab", type=int, default=5000)
    parser.add_argument("--context", type=int, default=2)
    parser.add_argument("--negative", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--logn", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    torch.set_num_threads(args.threads)

    dataset = Word2VecDataset(args.data, args.vocab, args.context, args.negative)
    dataloader = DataLoader(
        dataset,
        args.batch,
        shuffle=True,
        collate_fn=dataset.get_collate(),
        pin_memory=True,
        num_workers=args.threads,
    )
    actual_vocab_size = len(dataset.i2w)
    model = Word2Vec(actual_vocab_size, args.emb).to(device)
    optimizer = SGD(model.parameters(), args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))

    pbar = tqdm(total=len(dataloader) * args.epochs)

    try:
        iteration = 0
        avg_loss = 0
        for epoch in range(args.epochs):
            for batch in dataloader:
                center, context, negative = [t.to(device) for t in batch]

                optimizer.zero_grad()
                loss = model(center, context, negative)
                loss.backward()
                optimizer.step()
                scheduler.step()

                pbar.update()
                avg_loss += loss.detach().item()
                iteration += 1

                if iteration % args.logn == 0:
                    logger.info(f"Epoch {epoch}: {avg_loss / args.logn}")
                    avg_loss = 0
    except KeyboardInterrupt:
        logger.info("Forced stop.")

    weights = model.center_embeddings.weight.cpu().detach().numpy()
    np.save(args.output, weights)
    np.save(args.output + "-vocab", dataset.i2w)
