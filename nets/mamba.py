import jax
import jax.numpy as jnp
import equinox
from typing import Sequence, Tuple

class MambaBlock(equinox.Module):
    conv1: equinox.nn.Conv2d
    conv2: equinox.nn.Conv2d
    norm: equinox.nn.BatchNorm
    gate: equinox.nn.Linear
    input_proj: equinox.nn.Linear
    state_proj: equinox.nn.Linear
    output_proj: equinox.nn.Linear

    def __init__(self, in_ch: int, hidden_ch: int, *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.conv1 = equinox.nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1, key=k1)
        self.conv2 = equinox.nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, key=k2)
        self.norm = equinox.nn.BatchNorm(hidden_ch, axis_name="batch")
        self.gate = equinox.nn.Linear(hidden_ch, hidden_ch, key=k3)
        self.input_proj = equinox.nn.Linear(hidden_ch, hidden_ch, key=k4)
        self.state_proj = equinox.nn.Linear(hidden_ch, hidden_ch, key=k5)
        self.output_proj = equinox.nn.Linear(hidden_ch, in_ch, key=jax.random.split(k5)[0])

    def __call__(self, x, *, key=None):
        skip = x
        x = self.conv1(x)
        x = self.norm(x)
        x = jax.nn.silu(x)
        
        b, c, h, w = x.shape
        x_flat = x.reshape(b, c, -1).transpose(0, 2, 1)  # (b, seq, c)
        gate = jax.nn.sigmoid(self.gate(x_flat))
        input_proj = self.input_proj(x_flat)
        state = jnp.zeros_like(input_proj[:, 0:1, :]) 
        outputs = []
        
        for t in range(x_flat.shape[1]):
            state = state * 0.9 + input_proj[:, t:t+1, :]
            state_proj = self.state_proj(state)
            outputs.append(gate[:, t:t+1, :] * state_proj)
        
        x_rec = jnp.concatenate(outputs, axis=1).transpose(0, 2, 1).reshape(b, c, h, w)
        x = self.output_proj(x_rec.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        return skip + x

class MambaClassifier(equinox.Module):
    initial_conv: equinox.nn.Conv2d
    mamba_blocks: Sequence[MambaBlock]
    fc: equinox.nn.Linear
    dropout: equinox.nn.Dropout

    def __init__(self, image_size: int, in_ch: int, num_classes: int, *, key):
        k_initial, k_mamba, k_fc, k_drop = jax.random.split(key, 4)
        
        self.initial_conv = equinox.nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, key=k_initial)
        
        channels = [64, 128, 256]
        blocks = []
        k = k_mamba
        for i in range(3):
            k, subk = jax.random.split(k)
            blocks.append(MambaBlock(channels[i], channels[i] * 2, key=subk))
        self.mamba_blocks = tuple(blocks)
        
        spatial_size = image_size // (2 **3)  
        self.fc = equinox.nn.Linear(channels[-1] * spatial_size * spatial_size, num_classes, key=k_fc)
        self.dropout = equinox.nn.Dropout(p=0.5, key=k_drop)

    def __call__(self, x, *, key=None, train: bool = True):
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        x = self.initial_conv(x)
        x = jax.nn.relu(x)
        x = jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max,
            window_dimensions=(1, 1, 2, 2),
            window_strides=(1, 1, 2, 2),
            padding="VALID"
        )
        
        for block in self.mamba_blocks:
            x = block(x, key=key)
            x = jax.lax.reduce_window(
                x, -jnp.inf, jax.lax.max,
                window_dimensions=(1, 1, 2, 2),
                window_strides=(1, 1, 2, 2),
                padding="VALID"
            )
        
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x, key=key, inference=not train)
        logits = self.fc(x)
        return logits
