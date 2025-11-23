import os
import sys

# Add scripts to path
sys.path.append('./scripts')

# Install dependencies if needed (run once)
os.system('pip install equinox kaggle jax optax matplotlib pillow numpy')

from scripts.download import download_and_inspect_data
from scripts.preprocess import preprocess_folder, check_unique_sizes
from scripts.augment import augment_folder
from scripts.train import train_model

def main():
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
    model = train_model(data_dir, image_size=64, num_classes=9, epochs=10)
    '''
    print("Pipeline completed.")

if __name__ == "__main__":
    main()