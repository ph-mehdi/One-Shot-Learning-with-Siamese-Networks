# One-shot face matching with Siamese networks

An executable PyTorch study of **face verification** with limited images per person. The same image encoder is trained with either contrastive loss (image pairs) or triplet loss (anchor, positive, negative). This is an experimental matching pipeline, not a deployed identification or access-control system.

## What is in this repository

- `face_siamese.py` — data discovery, identity-level split, pair/triplet sampling, grayscale encoder, training, validation, and checkpointing.
- `One Shot Learning with Siamese Networks.ipynb` — a short notebook that runs the same implementation. The script is the source of truth.
- `tests/test_face_siamese.py` — synthetic-data checks for identity isolation, pair/triplet shapes and gradients, and a one-epoch training run for each loss.

## Dataset

The original exploration used the [AT&T face dataset](https://www.kaggle.com/datasets/kasikrit/att-database-of-faces). Obtain it under its own terms and place the images in **one folder per person**. The loader supports `.pgm`, `.png`, `.jpg`, and `.jpeg`.

```text
data/att_faces/
  s1/
    1.pgm
    2.pgm
    ...
  s2/
    1.pgm
    2.pgm
    ...
```

At least four person folders are required, with at least two images in each. The dataset is not redistributed here. The code never moves or rewrites source images. It uses a seeded **identity-level** train/validation split: images of the same person cannot appear on both sides. This measures matching on held-out identities. The validation split is small on AT&T, so a single split is not a reliable benchmark.

## Run

Use Python 3.10 or newer. In a virtual environment:

```bash
python -m pip install -r requirements.txt
python face_siamese.py --data data/att_faces --loss contrastive --epochs 10 --checkpoint checkpoints/contrastive.pt
python face_siamese.py --data data/att_faces --loss triplet --epochs 10 --checkpoint checkpoints/triplet.pt
```

The default image size is 100 × 100 pixels. The script selects CUDA when available and otherwise uses CPU. Run `python face_siamese.py --help` for the remaining options. The notebook expects the same `data/att_faces` folder and calls this script's `main` function.

Each epoch prints train and validation **loss**. The checkpoint contains the best validation-loss model weights, loss type, image size, seed, and the names of the train/validation identities. No accuracy, F1, or deployment claim is implied by a lower loss. For a benchmark, evaluate on a separate fixed set of identity-disjoint pairs and report threshold selection and uncertainty.

## Check the code

```bash
python -m unittest discover -s tests -v
```

The automated checks generate tiny synthetic images; the AT&T dataset is not required to run them.

## Changes from the original notebook

The exploratory notebook had a syntax error in its test cell, an outdated `Variable(..., volatile=True)` call, random validation augmentations, a platform-dependent path split, and a reference to an absent model checkpoint. The code now has one training entry point, deterministic validation sampling, identity isolation, explicit dataset validation, and a checkpoint written by the training command. Historical notebook outputs were removed because they cannot verify the revised implementation.
