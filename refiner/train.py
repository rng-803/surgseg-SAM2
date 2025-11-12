"""Training loop skeleton for the tiny refiner head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from refiner.models import TinyRefiner, TinyRefinerConfig

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_index(path: Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No entries found in {path}")
    return records


class RefinerDataset(Dataset):
    def __init__(self, index_path: Path, project_root: Path, priors_dir: Path | None = None, num_classes: int = 2):
        self.records = _load_index(index_path)
        self.project_root = project_root
        self.priors_dir = priors_dir
        self.num_classes = num_classes
        self.to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        from PIL import Image

        image = Image.open(self.project_root / rel_path).convert("RGB")
        return self.to_tensor(image)

    def _load_mask(self, rel_path: str) -> torch.Tensor:
        from PIL import Image

        mask = Image.open(self.project_root / rel_path)
        arr = np.array(mask, dtype=np.int64)
        if arr.ndim == 3:
            arr = arr[..., 0]
        return torch.from_numpy(arr)

    def _load_prior(self, image_rel_path: str, mask_tensor: torch.Tensor) -> torch.Tensor:
        if self.priors_dir is None:
            return F.one_hot(mask_tensor.clamp(max=self.num_classes - 1), num_classes=self.num_classes).permute(2, 0, 1).float()
        image_rel = Path(image_rel_path)
        candidates = [
            self.priors_dir / image_rel.with_suffix(".npy"),
            self.priors_dir / image_rel.name.replace(image_rel.suffix, ".npy"),
        ]
        for candidate in candidates:
            if candidate.exists():
                prior = np.load(candidate)
                return torch.from_numpy(prior).float()
        # Fall back to mask-derived one-hot if precomputed logits are missing.
        return F.one_hot(mask_tensor.clamp(max=self.num_classes - 1), num_classes=self.num_classes).permute(2, 0, 1).float()

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = self._load_image(record["image_path"])
        mask = self._load_mask(record["mask_path"])
        prior = self._load_prior(record["image_path"], mask)
        inputs = torch.cat([image, prior], dim=0)
        return {"inputs": inputs, "mask": mask}


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (probs * target_oh).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + target_oh.sum(dim=(0, 2, 3))
    dice = 1 - ((2 * intersection + eps) / (union + eps))
    return dice.mean()


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = RefinerDataset(args.index_path, args.project_root, args.priors_dir, args.num_classes)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    model = TinyRefiner(
        TinyRefinerConfig(
            in_channels=3 + args.num_classes,
            num_classes=args.num_classes,
            width_mult=args.width_mult,
            num_stages=args.num_stages,
            hidden_channels=args.hidden_channels,
            boundary_head=not args.no_boundary_head,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ce_loss = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            target = batch["mask"].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(inputs)
                logits = outputs["logits"]
                loss = ce_loss(logits, target)
                if args.dice_weight > 0:
                    loss = loss + args.dice_weight * dice_loss(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{args.epochs} - loss: {epoch_loss:.4f}")
        if (epoch + 1) % args.checkpoint_every == 0:
            ckpt_path = args.checkpoint_dir / f"refiner_epoch_{epoch+1:03d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch + 1}, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=Path("data/dataset.jsonl"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--priors-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("refiner/checkpoints"))
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--no-boundary-head", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
