import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from face_siamese import (
    FaceEncoder, FaceSamples, batch_loss, find_identities, main, split_identities,
)


class FaceSiameseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for person in range(6):
            folder = self.root / f"person_{person}"
            folder.mkdir()
            for image in range(3):
                pixels = np.full((24, 24), person * 30 + image, dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"image_{image}.png")

    def test_split_has_no_overlapping_identities(self):
        identities = find_identities(self.root)
        train, valid = split_identities(identities, seed=42)
        self.assertTrue(set(train).isdisjoint(valid))
        self.assertEqual(set(train) | set(valid), set(identities))
        self.assertEqual(list(valid), list(split_identities(identities, 42)[1]))

    def test_pair_and_triplet_have_valid_shapes_and_gradients(self):
        identities, _ = split_identities(find_identities(self.root), 42)
        model = FaceEncoder()
        for kind in ("contrastive", "triplet"):
            sample = FaceSamples(identities, kind, image_size=32, seed=42)[0]
            for image in sample[:2]:
                self.assertEqual(tuple(image.shape), (1, 32, 32))
            tensors = [part.unsqueeze(0) for part in sample]
            loss = batch_loss(model, tensors, kind, torch.device("cpu"))
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))
            model.zero_grad(set_to_none=True)

    def test_training_writes_loadable_checkpoint_for_both_losses(self):
        for kind in ("contrastive", "triplet"):
            checkpoint = self.root / f"{kind}.pt"
            main(["--data", str(self.root), "--loss", kind,
                  "--epochs", "1", "--batch-size", "4",
                  "--image-size", "32", "--checkpoint", str(checkpoint)])
            self.assertTrue(checkpoint.is_file())
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(saved["loss"], kind)
            self.assertTrue(set(saved["train_identities"]).isdisjoint(
                saved["validation_identities"]))
            FaceEncoder().load_state_dict(saved["model_state"])


if __name__ == "__main__":
    unittest.main()
