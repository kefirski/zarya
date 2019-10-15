# Hierarchical embeddings

Implementation of [Poincaré Embeddings for Learning Hierarchical Representations](https://arxiv.org/pdf/1705.08039.pdf) paper.

### How to run model:

1. Make sure that library is installed:
```sh
# In zarya root folder
python setup.py install
```
2. Preprocess dataset: train file must contain hierarchical pairs separated by comma
3. Train model
```sh
python train.py --source data.csv --out dump
```
4. Visualize results
```sh
python visualize.py --weight dump.npy --vocab dump_vocab.npy --out plot.png
```

