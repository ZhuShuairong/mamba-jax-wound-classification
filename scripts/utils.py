import os
import PIL.Image
from typing import List, Tuple, Dict

def parse_label_from_name(fname: str) -> str:
    base, _ = os.path.splitext(fname)
    parts = base.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected filename format: {fname}")
    label_part = parts[1]
    if "_aug" in label_part:
        label_part = label_part.split("_aug", 1)[0]
    return label_part

def build_index_flat(root_dir: str, val_ratio: float = 0.2, seed: int = 0) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], Dict[str, int]]:
    import random
    all_files = [f for f in os.listdir(root_dir) if f.lower().endswith(".jpg")]
    labels = [parse_label_from_name(f) for f in all_files]
    class_names = sorted(set(labels))
    label2idx = {name: i for i, name in enumerate(class_names)}

    samples = []
    for fname, lbl in zip(all_files, labels):
        path = os.path.join(root_dir, fname)
        samples.append((path, label2idx[lbl]))

    random.Random(seed).shuffle(samples)
    n_total = len(samples)
    n_val = int(n_total * val_ratio)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    return train_samples, val_samples, label2idx