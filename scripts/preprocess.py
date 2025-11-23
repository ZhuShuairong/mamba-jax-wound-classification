import os
import PIL.Image
from typing import Tuple

def preprocess_folder(dst_dir: str, target_size: Tuple[int, int] = (224, 224), mode: str = "resize", extensions: Tuple[str, ...] = (".jpg",)):
    """
    "resize": simple resize, may distort aspect ratio
    "resize_pad": keep aspect ratio, pad with black
    "center_crop": center-crop then resize to target_size
    """
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir("./data/dataset"):
        if not fname.lower().endswith(extensions):
            continue
        src_path = os.path.join("./data/dataset", fname)
        dst_path = os.path.join(dst_dir, fname)
        try:
            img = PIL.Image.open(src_path).convert("RGB")
        except Exception as e:
            print("Error:", src_path, e)
            continue
        if mode == "resize":
            img = img.resize(target_size, PIL.Image.LANCZOS)
        elif mode == "resize_pad":
            img.thumbnail(target_size, PIL.Image.LANCZOS)
            new_img = PIL.Image.new("RGB", target_size, (0, 0, 0))
            w, h = img.size
            left = (target_size[0] - w) // 2
            top = (target_size[1] - h) // 2
            new_img.paste(img, (left, top))
            img = new_img
        elif mode == "center_crop":
            w, h = img.size
            min_side = min(w, h)
            left = (w - min_side) // 2
            top = (h - min_side) // 2
            right = left + min_side
            bottom = top + min_side
            img = img.crop((left, top, right, bottom))
            img = img.resize(target_size, PIL.Image.LANCZOS)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        img.save(dst_path, quality=95)

def check_unique_sizes():
    unique_sizes = set()
    for fname in os.listdir("./data/dataset"):
        if not fname.lower().endswith(".jpg"):
            continue
        fpath = os.path.join("./data/dataset", fname)
        try:
            with PIL.Image.open(fpath) as img:
                w, h = img.size
        except Exception as e:
            print("Error reading", fpath, e)
            continue
        unique_sizes.add((w, h))
    print("Number of unique sizes:", len(unique_sizes))
    print("Unique sizes:")
    for w, h in sorted(unique_sizes):
        print(f"{w}x{h}")