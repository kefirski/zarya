import argparse

import torch as t
import torch.nn as nn
from dataloader import Dataloader
from model import Model
from tensorboardX import SummaryWriter
from torch.optim import Adam

import zarya.nn as znn
from zarya.manifolds import PoincareBall
from zarya.optim import RSGD

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="clickbait")
    parser.add_argument("--num-iterations", type=int, default=70000, metavar="NI")
    parser.add_argument("--batch-size", type=int, default=2, metavar="BS")
    parser.add_argument("--num-threads", type=int, default=4, metavar="NT")
    parser.add_argument("--embedding-size", type=int, default=5, metavar="ES")
    parser.add_argument("--hidden-size", type=int, default=15, metavar="HS")
    parser.add_argument("--tensorboard", type=str, default="default_tb", metavar="TB")
    parser.add_argument("--data", type=str, default="./data/", metavar="DP")
    args = parser.parse_args()

    device = "cpu"

    writer = SummaryWriter(args.tensorboard)

    t.set_num_threads(args.num_threads)
    loader = Dataloader(args.data)

    manifold = PoincareBall()

    model = Model(
        vocab_size=len(loader.idx_to_char),
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        manifold=manifold,
    ).cuda()
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    euc_optimizer = Adam(
        [p for p in model.parameters() if not isinstance(p, znn.Parameter)], lr=0.0002
    )
    hyp_optimizer = RSGD(
        [p for p in model.parameters() if isinstance(p, znn.Parameter)],
        manifold=manifold,
        lr=0.001,
    )

    for i in range(args.num_iterations):

        euc_optimizer.zero_grad()
        hyp_optimizer.zero_grad()

        model.train()

        input, target = loader.next_batch(args.batch_size, 200, "train", device)
        nll = model(input, target).mean()
        nll.backward()

        euc_optimizer.step()
        hyp_optimizer.step()

        model.eval()

        if i % 10 == 0:
            input, target = loader.next_batch(args.batch_size, 200, "valid", device)
            with t.no_grad():
                valid_bpc = model(input, target, 2).mean()

                writer.add_scalar("train nll", nll.cpu(), i)
                writer.add_scalar("valid bpc", valid_bpc.cpu(), i)
                print("i {}, train {} valid {}".format(i, nll.item(), valid_bpc.item()))
                print("_________")
