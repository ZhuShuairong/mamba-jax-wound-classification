import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scripts.model import Baseline
from scripts.data_loader import create_data_loader
import pickle

def evaluate_model(model: Baseline, 
                  test_samples: List[Tuple[str, int]], 
                  label2idx: dict,
                  batch_size: int = 32,
                  image_size: int = 64):
    all_preds = []
    all_labels = []
    
    test_loader = create_data_loader(test_samples, batch_size, image_size, shuffle=False)
    for x_batch, y_batch in test_loader:
        logits = model(x_batch, key=None, train=False)
        preds = logits.argmax(axis=-1)
        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch.tolist())
    
    idx2label = {v: k for k, v in label2idx.items()}
    class_names = [idx2label[i] for i in range(len(label2idx))]
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(10, 8))
    disp.plot(xticks_rotation=45)
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    return {
        "accuracy": (np.array(all_preds) == np.array(all_labels)).mean(),
        "classification_report": classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    }

def save_model(model: Baseline, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {save_path}")

def load_model(load_path: str) -> Baseline:
    with open(load_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {load_path}")
    return model
