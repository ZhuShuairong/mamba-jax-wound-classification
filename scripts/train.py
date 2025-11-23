import jax
import jax.numpy as jnp
import equinox
import optax
from scripts.model import Baseline
from scripts.data_loader import create_data_loader
from scripts.utils import build_index_flat

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, one_hot).mean()
    return loss

def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    preds = logits.argmax(axis=-1)
    return (preds == labels).mean()

@equinox.filter_jit
def train_step(model: Baseline, optimizer, opt_state, x: jnp.ndarray, y: jnp.ndarray, key):
    def loss_fn(m):
        logits = m(x, key=key, train=True)
        loss = cross_entropy_loss(logits, y)
        return loss, logits

    (loss, logits), grads = equinox.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = equinox.apply_updates(model, updates)
    acc = compute_accuracy(logits, y)
    return model, opt_state, loss, acc

@equinox.filter_jit
def eval_step(model: Baseline, x: jnp.ndarray, y: jnp.ndarray):
    logits = model(x, key=None, train=False)
    loss = cross_entropy_loss(logits, y)
    acc = compute_accuracy(logits, y)
    return loss, acc

def train_model(data_dir: str, image_size: int = 64, num_classes: int = 9, epochs: int = 10, batch_size: int = 32):
    learning_rate = 1e-3
    weight_decay = 1e-4
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)

    key = jax.random.PRNGKey(42)

    model = Baseline(
        image_size=image_size,
        in_ch=3,
        num_classes=num_classes,
        key=key,
    )

    opt_state = optimizer.init(equinox.filter(model, equinox.is_inexact_array))

    train_samples, val_samples, label2idx = build_index_flat(data_dir, val_ratio=0.2, seed=42)
    num_classes = len(label2idx)
    print("classes:", label2idx)
    print("train samples:", len(train_samples))
    print("val samples:", len(val_samples))

    for epoch in range(1, epochs + 1):
        train_losses = []
        train_accs = []

        train_loader = create_data_loader(train_samples, batch_size=batch_size)
        for x_batch, y_batch in train_loader:
            key, subkey = jax.random.split(key)
            model, opt_state, loss, acc = train_step(model, optimizer, opt_state, x_batch, y_batch, subkey)
            train_losses.append(loss)
            train_accs.append(acc)

        train_loss = float(jnp.mean(jnp.array(train_losses)))
        train_acc = float(jnp.mean(jnp.array(train_accs)))

        val_losses = []
        val_accs = []
        val_loader = create_data_loader(val_samples, batch_size=batch_size)
        for x_batch, y_batch in val_loader:
            loss, acc = eval_step(model, x_batch, y_batch)
            val_losses.append(loss)
            val_accs.append(acc)

        val_loss = float(jnp.mean(jnp.array(val_losses)))
        val_acc = float(jnp.mean(jnp.array(val_accs)))

        print(f"epoch {epoch} train loss {train_loss:.4f} acc {train_acc:.4f} val loss {val_loss:.4f} acc {val_acc:.4f}")

    return model