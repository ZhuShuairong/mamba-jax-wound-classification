import os
import kaggle
import PIL.Image

def download_and_inspect_data():
    # Set Kaggle config directory to keys folder
    os.environ['KAGGLE_CONFIG_DIR'] = './keys'
    
    # Authenticate with Kaggle
    kaggle.api.authenticate()

    # Download dataset if not exists
    data_path = "./data/Wound_dataset copy"
    if not os.path.exists(data_path):
        kaggle.api.dataset_download_files("ibrahimfateen/wound-classification", path="./data", unzip=True)

    # Inspect data
    label_counts = {}
    size_counts = {}
    image_info = []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            if not fname.lower().endswith((".jpg")):
                continue
            fpath = os.path.join(label_path, fname)
            try:
                with PIL.Image.open(fpath) as img:
                    w, h = img.size
            except Exception as e:
                print("Error reading", fpath, e)
                continue
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1
            key = (w, h)
            if key not in size_counts:
                size_counts[key] = 0
            size_counts[key] += 1
            image_info.append({
                "path": fpath,
                "label": label,
                "width": w,
                "height": h,
            })
    print("Label counts:", label_counts)
    print("Size counts (first few):", list(size_counts.items())[:10])

    # Rename and copy to dataset folder
    os.makedirs("./data/dataset", exist_ok=True)
    index = 1
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            if not fname.lower().endswith(".jpg"):
                continue
            src_path = os.path.join(label_path, fname)
            safe_label = label.replace(" ", "_")
            new_name = f"{index:06d}_{safe_label}.jpg"
            dst_path = os.path.join("./data/dataset", new_name)
            with open(src_path, "rb") as fsrc:
                data = fsrc.read()
            with open(dst_path, "wb") as fdst:
                fdst.write(data)
            index += 1