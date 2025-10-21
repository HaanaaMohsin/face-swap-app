import os
import argparse
from typing import List

import torch
from torch import nn

from .data import load_image_for_inference
from .model import build_model, load_checkpoint


def softmax_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


def predict_image(
    image_path: str,
    checkpoint_path: str,
    device: str = "cpu",
) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_idx = ckpt.get("class_to_idx")
    idx_to_class = ckpt.get("idx_to_class")
    image_size = ckpt.get("image_size", 224)
    model_name = ckpt.get("model_name", "swin_tiny_patch4_window7_224")
    num_classes = ckpt.get("num_classes", len(idx_to_class) if idx_to_class else 2)

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.to(device)
    model.eval()

    _ = load_checkpoint(model, checkpoint_path, map_location=device)

    image_tensor = load_image_for_inference(image_path, image_size=image_size).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = softmax_logits_to_probs(logits)[0]
        top_prob, top_idx = probs.max(dim=0)

    classes = [idx_to_class[i] for i in range(num_classes)] if idx_to_class else [str(i) for i in range(num_classes)]
    print(f"Image: {image_path}")
    for i, c in enumerate(classes):
        print(f"  {c:20s}: {probs[i].item():.4f}")
    print(f"Predicted: {classes[top_idx.item()]} ({top_prob.item():.4f})")


def predict_folder(folder_path: str, checkpoint_path: str, device: str = "cpu") -> None:
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    for fp in sorted(files):
        predict_image(fp, checkpoint_path, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Eye Disease Classification using Swin Transformer")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pth")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to an image file")
    group.add_argument("--folder", type=str, help="Path to a folder of images")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.image:
        predict_image(args.image, args.checkpoint, device=args.device)
    else:
        predict_folder(args.folder, args.checkpoint, device=args.device)
