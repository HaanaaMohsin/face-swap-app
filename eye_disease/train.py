import os
import argparse
import time
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from .data import create_dataloaders
from .model import build_model, save_checkpoint


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    preds = output.argmax(dim=1)
    correct = (preds == target).sum().item()
    return correct / target.size(0)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy(outputs.detach(), labels) * batch_size
        total += batch_size

    return running_loss / total, running_acc / total


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy(outputs, labels) * batch_size
            total += batch_size

    return running_loss / total, running_acc / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Swin Transformer for Eye Disease Classification")
    parser.add_argument("data_dir", type=str, help="Path to dataset (ImageFolder or with train/val folders)")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--model-name", type=str, default="swin_tiny_patch4_window7_224", help="timm model name")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not use pretrained weights")
    parser.add_argument("--balance", action="store_true", help="Use class-balanced sampling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze all layers except classifier head")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = select_device(prefer_gpu=True)
    print(f"Using device: {device}")

    train_loader, val_loader, idx_to_class, class_to_idx = create_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balance=args.balance,
    )

    num_classes = len(idx_to_class)
    model = build_model(args.model_name, num_classes=num_classes, pretrained=not args.no_pretrained)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if "head" in name or "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=not args.no_amp and device.type == "cuda")

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "model_best.pth")

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp=(not args.no_amp and device.type == "cuda"))
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start
        print(f"Epoch {epoch:03d}/{args.epochs} | Train: loss {train_loss:.4f}, acc {train_acc:.4f} | Val: loss {val_loss:.4f}, acc {val_acc:.4f} | {elapsed:.1f}s")

        # Save last checkpoint
        last_ckpt_path = os.path.join(args.output_dir, "model_last.pth")
        save_checkpoint(
            last_ckpt_path,
            model,
            optimizer,
            epoch,
            best_val_acc,
            class_to_idx,
            idx_to_class,
            args.image_size,
            args.model_name,
            num_classes,
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                best_ckpt_path,
                model,
                optimizer,
                epoch,
                best_val_acc,
                class_to_idx,
                idx_to_class,
                args.image_size,
                args.model_name,
                num_classes,
            )
            print(f"Saved new best: {best_ckpt_path} (acc {best_val_acc:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    main()
