"""Reproducible one-shot face matching with contrastive or triplet loss."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".pgm"}


def find_identities(root: Path) -> dict[str, list[Path]]:
    """Return sorted images by identity; each identity needs two distinct images."""
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    identities = {}
    for folder in sorted(root.iterdir()):
        if folder.is_dir():
            images = sorted(p for p in folder.iterdir()
                            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if images:
                if len(images) < 2:
                    raise ValueError(f"Identity {folder.name!r} needs at least two images")
                identities[folder.name] = images
    if len(identities) < 4:
        raise ValueError("At least four identity folders with two images each are required")
    return identities


def split_identities(identities: dict[str, list[Path]], seed: int,
                     validation_fraction: float = 0.2):
    """Keep all images of each person on one side of the split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    names = sorted(identities)
    if len(names) < 4:
        raise ValueError("At least four identities are required")
    random.Random(seed).shuffle(names)
    n_valid = max(2, min(len(names) - 2, round(len(names) * validation_fraction)))
    valid_names = set(names[:n_valid])
    train = {name: identities[name] for name in sorted(identities) if name not in valid_names}
    valid = {name: identities[name] for name in sorted(valid_names)}
    return train, valid


class FaceSamples(Dataset):
    def __init__(self, identities: dict[str, list[Path]], kind: str,
                 image_size: int = 100, training: bool = False, seed: int = 42):
        if kind not in {"contrastive", "triplet"}:
            raise ValueError("kind must be contrastive or triplet")
        if len(identities) < 2 or any(len(paths) < 2 for paths in identities.values()):
            raise ValueError("Each split needs two identities with two images per identity")
        if image_size < 16:
            raise ValueError("image_size must be at least 16")
        self.identities = identities
        self.kind = kind
        self.image_size = image_size
        self.training = training
        self.seed = seed
        self.names = sorted(identities)
        self.anchors = [(name, path) for name in self.names for path in identities[name]]

    def __len__(self):
        return len(self.anchors)

    def _image(self, path: Path, rng: random.Random) -> torch.Tensor:
        with Image.open(path) as raw:
            image = raw.convert("L").resize(
                (self.image_size, self.image_size), Image.Resampling.BILINEAR)
            if self.training and rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def __getitem__(self, index: int):
        rng = random if self.training else random.Random(self.seed + index)
        identity, anchor = self.anchors[index]
        positive = rng.choice([p for p in self.identities[identity] if p != anchor])
        other_name = rng.choice([name for name in self.names if name != identity])
        negative = rng.choice(self.identities[other_name])
        if self.kind == "triplet":
            return (self._image(anchor, rng), self._image(positive, rng),
                    self._image(negative, rng))
        different = bool(rng.randrange(2))
        partner = negative if different else positive
        return (self._image(anchor, rng), self._image(partner, rng),
                torch.tensor(float(different), dtype=torch.float32))


class FaceEncoder(nn.Module):
    """Small grayscale encoder shared across the two or three inputs."""

    def __init__(self, embedding_size: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, embedding_size),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.features(image)), dim=1)


def batch_loss(model: FaceEncoder, batch, kind: str, device: torch.device):
    if kind == "contrastive":
        first, second, different = (part.to(device) for part in batch)
        distance = F.pairwise_distance(model(first), model(second))
        return ((1 - different) * distance.square() +
                different * F.relu(1.0 - distance).square()).mean()
    anchor, positive, negative = (part.to(device) for part in batch)
    return F.triplet_margin_loss(model(anchor), model(positive), model(negative),
                                 margin=0.2, reduction="mean")


def run_epoch(model: FaceEncoder, loader: DataLoader, kind: str,
              device: torch.device, optimizer=None) -> float:
    model.train(optimizer is not None)
    total = 0.0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, batch, kind, device)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total += loss.item() * batch[0].shape[0]
    return total / len(loader.dataset)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path,
                        help="Directory with one subdirectory per identity")
    parser.add_argument("--loss", choices=("contrastive", "triplet"),
                        default="contrastive")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    identities = find_identities(args.data)
    train_ids, valid_ids = split_identities(identities, args.seed)
    print(f"Identity split: {len(train_ids)} train / {len(valid_ids)} validation")
    train = FaceSamples(train_ids, args.loss, args.image_size, training=True,
                        seed=args.seed)
    valid = FaceSamples(valid_ids, args.loss, args.image_size, seed=args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)
    valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False,
                              num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, args.loss, device, optimizer)
        valid_loss = run_epoch(model, valid_loader, args.loss, device)
        print(f"epoch {epoch:02d}: train loss {train_loss:.4f}; "
              f"validation loss {valid_loss:.4f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": best_state, "loss": args.loss,
                        "image_size": args.image_size, "seed": args.seed,
                        "train_identities": sorted(train_ids),
                        "validation_identities": sorted(valid_ids)}, args.checkpoint)
    print(f"Best validation loss: {best_loss:.4f}; checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
