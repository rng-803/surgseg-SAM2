"""Training loop skeleton for the tiny refiner head."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
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
    def __init__(
        self,
        index_path: Path,
        project_root: Path,
        priors_dir: Path | None = None,
        num_classes: int = 3,
        input_size: Optional[int] = None,
    ):
        self.records = _load_index(index_path)
        self.project_root = project_root
        self.priors_dir = priors_dir
        self.num_classes = num_classes
        self.input_size = input_size
        tfms = []
        if input_size is not None:
            tfms.append(transforms.Resize((input_size, input_size)))
        tfms.extend([transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        self.to_tensor = transforms.Compose(tfms)

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rel_path: str) -> torch.Tensor:
        image = Image.open(self.project_root / rel_path).convert("RGB")
        return self.to_tensor(image)

    def _load_mask(self, rel_path: str) -> torch.Tensor:
        mask = Image.open(self.project_root / rel_path)
        if self.input_size is not None:
            mask = mask.resize((self.input_size, self.input_size), resample=Image.NEAREST)
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
                prior_tensor = torch.from_numpy(prior).float()
                if self.input_size is not None:
                    prior_tensor = F.interpolate(
                        prior_tensor.unsqueeze(0),
                        size=(self.input_size, self.input_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                    denom = prior_tensor.sum(dim=0, keepdim=True)
                    prior_tensor = prior_tensor / torch.clamp(denom, min=1e-6)
                return prior_tensor
        # Fall back to mask-derived one-hot if precomputed logits are missing.
        one_hot = (
            F.one_hot(mask_tensor.clamp(max=self.num_classes - 1), num_classes=self.num_classes)
            .permute(2, 0, 1)
            .float()
        )
        if self.input_size is not None:
            one_hot = F.interpolate(
                one_hot.unsqueeze(0),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            denom = one_hot.sum(dim=0, keepdim=True)
            one_hot = one_hot / torch.clamp(denom, min=1e-6)
        return one_hot

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = self._load_image(record["image_path"])
        mask = self._load_mask(record["mask_path"])
        prior = self._load_prior(record["image_path"], mask)
        inputs = torch.cat([image, prior], dim=0)
        return {
            "inputs": inputs,
            "mask": mask,
            "image": image,
            "prior": prior,
            "image_path": record["image_path"],
        }


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    target_oh = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    intersection = (probs * target_oh).sum(dim=(0, 2, 3))
    union = probs.sum(dim=(0, 2, 3)) + target_oh.sum(dim=(0, 2, 3))
    dice = 1 - ((2 * intersection + eps) / (union + eps))
    return dice.mean()


BASE_PALETTE = np.array(
    [
        [0, 0, 0],
        [0, 176, 246],
        [234, 92, 92],
        [120, 195, 20],
        [255, 214, 10],
        [196, 147, 214],
    ],
    dtype=np.uint8,
)


def _build_palette(num_classes: int) -> np.ndarray:
    repeats = math.ceil(num_classes / len(BASE_PALETTE))
    palette = np.tile(BASE_PALETTE, (repeats, 1))
    return palette[:num_classes]


def _tensor_to_image(image: torch.Tensor) -> np.ndarray:
    img = image.detach().cpu().clone()
    for channel, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        img[channel] = img[channel] * std + mean
    img = img.clamp(0.0, 1.0).permute(1, 2, 0).mul(255.0).byte().numpy()
    return img


def _colorize_mask(mask: torch.Tensor, palette: np.ndarray) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy().astype(np.int64)
    mask_np = np.clip(mask_np, 0, palette.shape[0] - 1)
    return palette[mask_np]


def _blend(image: np.ndarray, mask_color: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    return np.clip(alpha * mask_color + (1 - alpha) * image, 0, 255).astype(np.uint8)


def save_visualizations(examples: List[Dict], vis_dir: Path, epoch: int, num_classes: int) -> None:
    if not examples:
        return
    vis_dir.mkdir(parents=True, exist_ok=True)
    palette = _build_palette(num_classes)
    for idx, sample in enumerate(examples):
        image = _tensor_to_image(sample["image"])
        pred_overlay = _blend(image, _colorize_mask(sample["pred"], palette))
        target_overlay = _blend(image, _colorize_mask(sample["mask"], palette))
        grid = np.concatenate([image, pred_overlay, target_overlay], axis=1)
        stem = Path(sample["image_path"]).stem
        out_path = vis_dir / f"epoch_{epoch:03d}_{idx:02d}_{stem}.png"
        Image.fromarray(grid).save(out_path)


def _compute_metrics(stats: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    eps = 1e-6
    intersection = stats["intersection"].cpu().numpy()
    pred = stats["pred"].cpu().numpy()
    target = stats["target"].cpu().numpy()
    dice = (2 * intersection) / np.clip(pred + target, eps, None)
    iou = intersection / np.clip(pred + target - intersection, eps, None)
    valid = target > 0
    dice_mean = dice[valid].mean() if valid.any() else dice.mean()
    iou_mean = iou[valid].mean() if valid.any() else iou.mean()
    return {
        "dice": dice,
        "iou": iou,
        "dice_mean": float(dice_mean),
        "iou_mean": float(iou_mean),
    }


def run_validation(
    model: nn.Module,
    dataloader: Optional[DataLoader],
    device: torch.device,
    ce_loss: nn.Module,
    dice_weight: float,
    num_classes: int,
    amp_enabled: bool,
    max_vis_samples: int,
) -> tuple[Optional[Dict[str, float]], List[Dict]]:
    if dataloader is None:
        return None, []
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_samples = 0
    stats = {
        "intersection": torch.zeros(num_classes, dtype=torch.double),
        "pred": torch.zeros(num_classes, dtype=torch.double),
        "target": torch.zeros(num_classes, dtype=torch.double),
    }
    vis_examples: List[Dict] = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(inputs)
                logits = outputs["logits"]
                loss = ce_loss(logits, target)
                if dice_weight > 0:
                    loss = loss + dice_weight * dice_loss(logits, target)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            preds = torch.argmax(logits, dim=1)
            for cls in range(num_classes):
                pred_mask = preds == cls
                target_mask = target == cls
                stats["intersection"][cls] += (pred_mask & target_mask).sum().double().cpu()
                stats["pred"][cls] += pred_mask.sum().double().cpu()
                stats["target"][cls] += target_mask.sum().double().cpu()
            if max_vis_samples > 0 and len(vis_examples) < max_vis_samples:
                take = min(max_vis_samples - len(vis_examples), inputs.size(0))
                for i in range(take):
                    vis_examples.append(
                        {
                            "image": batch["image"][i].clone(),
                            "pred": preds[i].detach().cpu(),
                            "mask": target[i].detach().cpu(),
                            "image_path": batch["image_path"][i],
                        }
                    )
    metrics = _compute_metrics(stats)
    metrics["loss"] = total_loss / max(1, total_samples)
    if was_training:
        model.train()
    return metrics, vis_examples


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    val_fraction = max(0.0, min(args.val_fraction, 0.9))
    base_dataset = RefinerDataset(
        args.index_path,
        args.project_root,
        args.priors_dir,
        args.num_classes,
        input_size=args.train_size,
    )
    val_dataset: Optional[Dataset] = None
    if args.val_index_path:
        val_dataset = RefinerDataset(
            args.val_index_path,
            args.project_root,
            args.priors_dir,
            args.num_classes,
            input_size=args.train_size,
        )
        train_dataset = base_dataset
    elif val_fraction > 0:
        val_size = max(1, int(len(base_dataset) * val_fraction))
        if 0 < val_size < len(base_dataset):
            lengths = [len(base_dataset) - val_size, val_size]
            train_dataset, val_dataset = random_split(
                base_dataset, lengths, generator=torch.Generator().manual_seed(args.seed)
            )
        else:
            train_dataset = base_dataset
    else:
        train_dataset = base_dataset
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=pin_memory,
        )
        if val_dataset is not None
        else None
    )
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
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    ce_loss = nn.CrossEntropyLoss()
    model.train()
    train_len = len(train_loader.dataset)
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch in train_loader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=args.amp):
                outputs = model(inputs)
                logits = outputs["logits"]
                loss = ce_loss(logits, target)
                if args.dice_weight > 0:
                    loss = loss + args.dice_weight * dice_loss(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / max(1, train_len)
        print(f"Epoch {epoch+1}/{args.epochs} - train loss: {epoch_loss:.4f}")
        should_validate = val_loader is not None and (
            (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1
        )
        if should_validate:
            metrics, vis_examples = run_validation(
                model,
                val_loader,
                device,
                ce_loss,
                args.dice_weight,
                args.num_classes,
                args.amp,
                args.vis_samples,
            )
            if metrics:
                dice_str = ", ".join(f"{d:.3f}" for d in metrics["dice"])
                iou_str = ", ".join(f"{iou:.3f}" for iou in metrics["iou"])
                print(
                    f"  Val loss: {metrics['loss']:.4f} | Dice: {metrics['dice_mean']:.3f} "
                    f"(per-class: {dice_str}) | IoU: {metrics['iou_mean']:.3f} "
                    f"(per-class: {iou_str})"
                )
            if vis_examples:
                save_visualizations(vis_examples, args.val_vis_dir, epoch + 1, args.num_classes)
        if (epoch + 1) % args.checkpoint_every == 0:
            ckpt_path = args.checkpoint_dir / f"refiner_epoch_{epoch+1:03d}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "epoch": epoch + 1}, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=Path("data/dataset.jsonl"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--priors-dir", type=Path, default=Path("sam2_prior/logits"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("refiner/checkpoints"))
    parser.add_argument("--val-index-path", type=Path, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--val-vis-dir", type=Path, default=Path("refiner/val_vis"))
    parser.add_argument("--train-size", type=int, default=None, help="Optional square resize for images/masks/priors")
    parser.add_argument("--vis-samples", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=3)
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
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
