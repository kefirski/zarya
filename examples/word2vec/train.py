import argparse
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Hyperbolic Word2Vec Trainer")

    args = parser.parse_args()
