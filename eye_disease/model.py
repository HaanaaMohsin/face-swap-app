from typing import Optional, Dict, Any

import torch
from torch import nn
import timm


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


essential_ckpt_keys = [
    "state_dict",
    "class_to_idx",
    "idx_to_class",
    "image_size",
    "model_name",
    "num_classes",
]


def load_checkpoint(model: nn.Module, checkpoint_path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location or "cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    metadata = {
        "class_to_idx": checkpoint.get("class_to_idx"),
        "idx_to_class": checkpoint.get("idx_to_class"),
        "image_size": checkpoint.get("image_size", 224),
        "model_name": checkpoint.get("model_name"),
        "num_classes": checkpoint.get("num_classes"),
        "epoch": checkpoint.get("epoch"),
        "best_acc": checkpoint.get("best_acc"),
    }
    return {"missing_keys": missing, "unexpected_keys": unexpected, "metadata": metadata}


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_acc: float,
    class_to_idx: Dict[str, int],
    idx_to_class: Dict[int, str],
    image_size: int,
    model_name: str,
    num_classes: int,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "best_acc": best_acc,
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "image_size": image_size,
            "model_name": model_name,
            "num_classes": num_classes,
        },
        path,
    )
