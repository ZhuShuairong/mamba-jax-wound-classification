import os
import sys
import jax
import pickle
from pathlib import Path

# Add scripts to path
sys.path.append('./scripts')
sys.path.append('./nets')  # 确保能导入Mamba模型

# Install dependencies if needed (run once)
os.system('pip install equinox kaggle jax optax matplotlib pillow numpy')

from scripts.download import download_and_inspect_data
from scripts.preprocess import preprocess_folder, check_unique_sizes
from scripts.augment import augment_folder
from scripts.train import train_model
from scripts.evaluate import evaluate_model  # 新增：评估模块
from scripts.utils import build_index_flat  # 复用数据索引函数
from nets.mamba import MambaClassifier  # 新增：Mamba模型

def main():
    # 配置参数
    IMAGE_SIZE = 64  # 可选64/128/256
    DATA_DIR = f"./data/augmented/dataset_{IMAGE_SIZE}_low"  # 对应预处理数据路径
    NUM_CLASSES = 10  # 伤口类别数
    EPOCHS = 15
    BATCH_SIZE = 32
    MODEL_SAVE_PATH = f"./models/mamba_wound_{IMAGE_SIZE}.pkl"  # 模型保存路径

    # Step 1: Download and inspect data
    print("Downloading and inspecting data...")
    download_and_inspect_data()

    # Step 2: Check unique sizes
    print("Checking unique image sizes...")
    check_unique_sizes()

    # Step 3: Preprocess images
    print("Preprocessing images...")
    preprocess_folder("./data/2baugmented/dataset_low", (64, 64))
    preprocess_folder("./data/2baugmented/dataset_mid", (128, 128))
    preprocess_folder("./data/2baugmented/dataset_hig", (256, 256))

    # Step 4: Augment images
    print("Augmenting images...")
    augment_folder(src_dir="./data/2baugmented/dataset_low",
                   dst_dir="./data/augmented/dataset_64_low",
                   target_size=(64, 64),
                   num_aug_per_image=2)
    augment_folder(src_dir="./data/2baugmented/dataset_mid",
                   dst_dir="./data/augmented/dataset_128_mid",
                   target_size=(128, 128),
                   num_aug_per_image=2)
    augment_folder(src_dir="./data/2baugmented/dataset_hig",
                   dst_dir="./data/augmented/dataset_256_hig",
                   target_size=(256, 256),
                   num_aug_per_image=2)
    '''
    # Step 5: Train model
    print("Training model...")
    data_dir = "./data/augmented/dataset_64_low"
    model = train_model(data_dir, image_size=64, num_classes=10, epochs=10)
    '''
    print("Pipeline completed.")

    # Step 6: Evaluate modele
    print("Evaluating model...")
    _, val_samples, label2idx = build_index_flat(DATA_DIR, val_ratio=0.2, seed=42)
    metrics = evaluate_model(
        model=model,
        samples=val_samples,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )
    print("Evaluation indicators:")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    print(f"Accuracy of each category: {metrics['per_class_accuracy']}")

    # Step 7: save model
    save_model(model, MODEL_SAVE_PATH)
    print("Pipeline completed.")
    

if __name__ == "__main__":
    main()
