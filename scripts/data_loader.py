import os
import jax.numpy as jnp
import numpy as np
import PIL.Image
from typing import List, Tuple, Iterator

def load_image(path: str) -> np.ndarray:
    with PIL.Image.open(path) as img:
        img = img.convert("RGB")
        return np.array(img) / 255.0  # Normalize to [0,1]

def create_data_loader(samples: List[Tuple[str, int]], batch_size: int = 32) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i+batch_size]
        images = []
        labels = []
        for path, label in batch_samples:
            img = load_image(path)
            images.append(img)
            labels.append(label)
        x_batch = jnp.array(images, dtype=jnp.float32)
        y_batch = jnp.array(labels, dtype=jnp.int32)
        yield x_batch, y_batch