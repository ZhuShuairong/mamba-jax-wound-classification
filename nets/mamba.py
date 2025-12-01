import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence


class MambaBlock(nn.Module):
    """
    简单残差块：LayerNorm → Conv → GELU → Dropout → Residual
    """
    in_channels: int
    out_channels: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        :param x: 输入特征 (N, H, W, C)
        :param train: 是否训练模式（控制 Dropout）
        :return: 残差块输出
        """
        # 残差分支
        skip = x
        if self.in_channels != self.out_channels:
            skip = nn.Conv(
                features=self.out_channels,
                kernel_size=(1, 1),
                padding="SAME",
                use_bias=False,
            )(skip)

        # 主分支
        out = nn.LayerNorm(epsilon=1e-6)(x)
        out = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
        )(out)
        out = nn.gelu(out)
        out = nn.Dropout(rate=self.dropout_rate)(
            out, deterministic=not train
        )

        return skip + out


class MambaClassifier(nn.Module):
    """
    Mamba 风格分类网络
    """
    image_size: int
    in_channels: int
    num_classes: int
    dropout_rate: float = 0.5
    filters: Sequence[int] = (64, 128, 256)

    def setup(self):
        # Stem：将输入映射到 filters[0]
        self.stem_conv = nn.Conv(
            features=self.filters[0],
            kernel_size=(7, 7),
            strides=(2, 2),
            padding="SAME",
            use_bias=False,
        )

        # Stage 下采样卷积
        self.down_convs = tuple(
            nn.Conv(
                features=self.filters[i],
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                use_bias=False,
            )
            for i in range(1, len(self.filters))
        )

        # Stage 内 MambaBlock（每个 stage 两个 block）
        blocks = []
        for c in self.filters:
            stage_blocks = (
                MambaBlock(c, c, dropout_rate=self.dropout_rate),
                MambaBlock(c, c, dropout_rate=self.dropout_rate),
            )
            blocks.append(stage_blocks)
        self.blocks = tuple(blocks)

        # 分类层
        self.classifier_dense = nn.Dense(self.num_classes)

    @nn.compact
    def __call__(self, x, train: bool = True):
        """
        :param x: 输入图像 (N, H, W, C)，H=W=image_size
        :param train: 是否训练模式
        :return: logits (N, num_classes)
        """
        # Stem
        x = self.stem_conv(x)
        x = nn.gelu(x)

        # Stage 0
        for block in self.blocks[0]:
            x = block(x, train=train)

        # Stage 1+
        for i in range(1, len(self.filters)):
            x = self.down_convs[i - 1](x)
            x = nn.gelu(x)
            for block in self.blocks[i]:
                x = block(x, train=train)

        # Global Average Pooling
        x = x.mean(axis=(1, 2))  # (N, C)

        # Dropout + 全连接分类
        x = nn.Dropout(rate=self.dropout_rate)(
            x, deterministic=not train
        )
        logits = self.classifier_dense(x)

        return logits


# ========== 测试代码 (JAX 版本) ==========
if __name__ == "__main__":
    # ---------------- 基本配置 ----------------
    image_size = 128
    in_channels = 3
    num_classes = 10
    batch_size = 2

    device = jax.devices()[0]
    print(f"Using device: {device}")

    # ---------------- 构建模型 ----------------
    model = MambaClassifier(
        image_size=image_size,
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=0.5,
        filters=(64, 128, 256),
    )
    print(model)

    # ---------------- 构造假数据 (NHWC) ----------------
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch_size, image_size, image_size, in_channels))
    print(f"\nInput shape: {x.shape}")

    # ---------------- 初始化参数 ----------------
    # 注意：train=False → Dropout 为 deterministic，不需要显式传 dropout rng
    variables = model.init(key, x, train=False)
    params = variables["params"]

    # 统计可训练参数数量
    def count_params(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return sum(p.size for p in leaves)

    print(f"\nTrainable parameters: {count_params(params)}")

    # ---------------- 前向传播（推理模式） ----------------
    logits = model.apply({"params": params}, x, train=False)
    print(f"\nLogits shape: {logits.shape}")
    assert logits.shape == (batch_size, num_classes), \
        f"Logits shape is wrong: {logits.shape}"

    # ---------------- 反向传播测试（训练模式） ----------------
    def loss_fn(p, x_in, y_in, rng):
        logits = model.apply(
            {"params": p},
            x_in,
            train=True,
            rngs={"dropout": rng},  # 只有训练时才需要给 Dropout rng
        )
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(y_in, num_classes=num_classes)
        return -(one_hot * log_probs).sum(axis=-1).mean()

    y = jax.random.randint(jax.random.PRNGKey(3), (batch_size,), 0, num_classes)
    rng = jax.random.PRNGKey(4)

    loss_value, grads = jax.value_and_grad(loss_fn)(
        params, x, y, rng
    )
    print(f"\nDummy loss: {float(loss_value):.6f}")

    max_grad = max(float(jnp.abs(g).max()) for g in jax.tree_util.tree_leaves(grads))
    print(f"Max grad: {max_grad:.6e}")

    print("\nAll tests finished. If no assertion error or exception occurred, the JAX/Flax MambaClassifier can now be initialized, forwarded, and reversed correctly.")
