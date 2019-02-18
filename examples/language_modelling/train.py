import argparse

import torch as t
import torch.nn as nn
from dataloader import Dataloader
from model import Model
from tensorboardX import SummaryWriter
from torch.optim import Adam

from zarya.optim import RSGD

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="clickbait")
    parser.add_argument("--num-iterations", type=int, default=70000, metavar="NI")
    parser.add_argument("--batch-size", type=int, default=64, metavar="BS")
    parser.add_argument("--num-threads", type=int, default=4, metavar="NT")
    parser.add_argument("--embedding-size", type=int, default=5, metavar="ES")
    parser.add_argument("--hidden-size", type=int, default=15, metavar="HS")
    parser.add_argument("--tensorboard", type=str, default="default_tb", metavar="TB")
    parser.add_argument("--data", type=str, default="./data/", metavar="DP")
    args = parser.parse_args()

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    writer = SummaryWriter(args.tensorboard)

    t.set_num_threads(args.num_threads)
    loader = Dataloader(args.data)

    model = Model(
        vocab_size=loader.sp.GetPieceSize(),
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
    )
    model.to(device)

    euc_optimizer = Adam(model.parameters(), lr=0.0002)
    hyp_optimizer = RSGD(model.hparameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for i in range(args.num_iterations):

        euc_optimizer.zero_grad()
        hyp_optimizer.zero_grad()

        model.train()

        input, target = loader.next_batch(args.batch_size, "train", device)
        nll = criterion(model(input)[0], target.view(-1))
        nll.backward()

        euc_optimizer.step()
        hyp_optimizer.step()

        model.eval()

        if i % 100 == 0:
            input, target = loader.next_batch(args.batch_size, "test", device)
            with t.no_grad():
                test_nll = criterion(model(input)[0], target.view(-1))

                writer.add_scalar("train loss", nll.cpu(), i)
                writer.add_scalar("test loss", test_nll.cpu(), i)
                print("i {}, train {} test {}".format(i, nll.item(), test_nll.item()))
                print("_________")

        if i % 20 == 0:
            with t.no_grad():
                generation = model.generate(1, device)
                print(loader.sp.DecodeIds(generation) + "\n")
