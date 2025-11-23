import jax
import jax.numpy as jnp
import equinox
from typing import Sequence

class ConvolutionalBlock(equinox.Module):
    conv: equinox.nn.Conv2d
    norm: equinox.nn.BatchNorm
    act: equinox.nn.Lambda

    def __init__(self, in_ch: int, out_ch: int, *, key):
        k1, k2 = jax.random.split(key, 2)
        self.conv = equinox.nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=3,
            padding=1,
            key=k1,
        )
        self.norm = equinox.nn.BatchNorm(out_ch, axis_name="batch", momentum=0.9, eps=1e-5)
        self.act = equinox.nn.Lambda(jax.nn.relu)

    def __call__(self, x, *, key=None):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class Baseline(equinox.Module):
    conv_blocks: Sequence[ConvolutionalBlock]
    fc1: equinox.nn.Linear
    fc2: equinox.nn.Linear
    fc3: equinox.nn.Linear
    dropout: equinox.nn.Dropout

    def __init__(self, image_size: int, in_ch: int, num_classes: int, *, key):
        k_conv, k_fc1, k_fc2, k_fc3, k_do = jax.random.split(key, 5)

        channels = [in_ch, 32, 64, 128, 256]
        blocks = []
        k = k_conv
        for i in range(4):
            k, subk = jax.random.split(k)
            blocks.append(ConvolutionalBlock(channels[i], channels[i+1], key=subk))
        self.conv_blocks = tuple(blocks)
        spatial = image_size // 16
        flattened_dim = channels[-1] * spatial * spatial

        self.fc1 = equinox.nn.Linear(flattened_dim, 256, key=k_fc1)
        self.fc2 = equinox.nn.Linear(256, 128, key=k_fc2)
        self.fc3 = equinox.nn.Linear(128, num_classes, key=k_fc3)
        self.dropout = equinox.nn.Dropout(p=0.5, key=k_do)

    def __call__(self, x, *, key=None, train: bool = True):
        x = jnp.transpose(x, (0, 3, 1, 2))

        for block in self.conv_blocks:
            x = block(x)
            x = jax.lax.reduce_window(
                x,
                -jnp.inf,
                jax.lax.max,
                window_dimensions=(1, 1, 2, 2),
                window_strides=(1, 1, 2, 2),
                padding="VALID",
            )

        x = x.reshape(x.shape[0], -1)

        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key, 2)

        x = self.fc1(x)
        x = jax.nn.relu(x)
        x = self.dropout(x, key=k1, inference=not train)

        x = self.fc2(x)
        x = jax.nn.relu(x)
        x = self.dropout(x, key=k2, inference=not train)

        logits = self.fc3(x)
        return logits