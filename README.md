# PassGAN

This repository contains code for the [_PassGAN: A Deep Learning Approach for Password Guessing_](https://arxiv.org/abs/1709.00440) paper. 

The model from PassGAN is taken from [_Improved Training of Wasserstein GANs_](https://arxiv.org/abs/1704.00028) and it is assumed that the authors of PassGAN used the [improved_wgan_training](https://github.com/igul222/improved_wgan_training) tensorflow implementation in their work. For this reason, I have modified that reference implementation in this repository to make it easy to train (`train.py`) and sample (`sample.py`) from. This repo contributes:

- A command-line interface
- A pretrained PassGAN models trained on the RockYou dataset

## Getting Started

```bash
# requires CUDA to be pre-installed
pip install -r requirements.txt
```

### Generating password samples

Use the pretrained model to generate 1,000,000 passwords, saving them to `gen_passwords.txt`.

```bash
python sample.py \
	--input-dir pretrained \
	--checkpoint pretrained/checkpoints/195000.ckpt \
	--output gen_passwords.txt \
	--batch-size 1024 \
	--num-samples 1000000
```

### Training your own models

Training a model on a large dataset (100MB+) can take several hours on a GTX 1080.

```bash
# download the rockyou training data
# contains 80% of the full rockyou passwords (with repeats)
# that are 10 characters or less
curl -L -o data/train.txt https://github.com/brannondorsey/PassGAN/releases/download/data/rockyou-train.txt

# train for 200000 iterations, saving checkpoints every 5000
# uses the default hyperparameters from the paper
python train.py --output-dir output --training-data data/train.txt
```

You are encouraged to train using your own password leaks and datasets. Some great places to find those include:

- [LinkedIn leak](https://hashes.org/download.php?hashlistId=68&type=hfound)(2.9GB, direct download)
- [Exploit.in torrent](https://thepiratebay.org/torrent/16016494/exploit.in) (10GB+, 800 million accounts. Infamous!)
- [Hashes.org](https://hashes.org/leaks.php): a shared password recovery site.
