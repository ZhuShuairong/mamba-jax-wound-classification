import os
import math
import random
import PIL.Image
import PIL.ImageEnhance
from typing import Tuple

def random_rotate(img: PIL.Image.Image, degrees: float = 20) -> PIL.Image.Image:
    angle = random.uniform(-degrees, degrees)
    return img.rotate(angle, resample=PIL.Image.BILINEAR)

def center_crop_resize(img: PIL.Image.Image, target_size: Tuple[int, int]) -> PIL.Image.Image:
    width, height = img.size
    min_side = min(width, height)
    left = (width - min_side) // 2
    top = (height - min_side) // 2
    right = left + min_side
    bottom = top + min_side
    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped.resize(target_size, PIL.Image.BILINEAR)

def random_resized_crop(img: PIL.Image.Image, target_size: Tuple[int, int], scale: Tuple[float, float] = (0.9, 1.0), ratio: Tuple[float, float] = (0.9, 1.1)) -> PIL.Image.Image:
    width, height = img.size
    area = width * height
    for _ in range(10):
        target_area = random.uniform(scale[0], scale[1]) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect = math.exp(random.uniform(*log_ratio))
        w = int(round(math.sqrt(target_area * aspect)))
        h = int(round(math.sqrt(target_area / aspect)))
        if 0 < w <= width and 0 < h <= height:
            left = random.randint(0, width - w)
            top = random.randint(0, height - h)
            img_cropped = img.crop((left, top, left + w, top + h))
            return img_cropped.resize(target_size, PIL.Image.BILINEAR)
    return center_crop_resize(img, target_size)

def random_brightness(img: PIL.Image.Image, max_delta: float = 0.2) -> PIL.Image.Image:
    factor = 1.0 + random.uniform(-max_delta, max_delta)
    enhancer = PIL.ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

def random_contrast(img: PIL.Image.Image, max_delta: float = 0.2) -> PIL.Image.Image:
    factor = 1.0 + random.uniform(-max_delta, max_delta)
    enhancer = PIL.ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def augment_image_pil(img: PIL.Image.Image, target_size: Tuple[int, int], degrees: float = 20) -> PIL.Image.Image:
    img = random_rotate(img, degrees=degrees)
    img = random_resized_crop(
        img,
        target_size=target_size,
        scale=(0.9, 1.0),
        ratio=(0.9, 1.1),
    )
    img = random_brightness(img, max_delta=0.2)
    img = random_contrast(img, max_delta=0.2)
    return img

def augment_folder(src_dir: str, dst_dir: str, target_size: Tuple[int, int], num_aug_per_image: int = 1):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if not fname.lower().endswith(".jpg"):
            continue
        src_path = os.path.join(src_dir, fname)
        try:
            img = PIL.Image.open(src_path).convert("RGB")
        except Exception as e:
            print("Error:", src_path, e)
            continue
        base_name, ext = os.path.splitext(fname)
        for i in range(num_aug_per_image):
            aug_img = augment_image_pil(img, target_size=target_size, degrees=20)
            new_name = f"{base_name}_aug{i:02d}{ext}"
            dst_path = os.path.join(dst_dir, new_name)
            aug_img.save(dst_path, quality=95)