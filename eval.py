import csv
import datetime
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.CNN_Mamba_CNN_Mamba_Enhanced.HybricMamba import HybricMamba
from models.vmamba.Vmamba_ultils import Super_Mamba

# =================================================================================
# CONFIGURATION
# =================================================================================
SEED = 42
PICTURE_SIZE = 32
BATCH_SIZE = 64
DATASET_NAME = "German"
DATASET_PATH = "/home/biu-linux/DeepLearning_Projects/DoAnNganh/dataset_reOrgan"
CHECKPOINT_PATH = "/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult/TFJ/SUPER_MAMBA_DEPT_4/German/Super_Mamba_dim_4_best.pth"
OUTPUT_DIR = "./evaluation_results"
LOG_FILE_PATH = "/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult/TFJ/SUPER_MAMBA_DEPT_4/German/training.log"

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.ppm')


# =================================================================================
# UTILS & DATASET CLASS
# =================================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]


def log_and_print(message, log_file):
    print(message)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


class TrafficSignDataset(Dataset):
    def __init__(self, root, dataset_name="German", csv_filename="Train.csv",
                 transform=None, samples=None, class_to_idx=None, shuffle_samples=True):
        self.root = root
        self.transform = transform
        self.dataset_name = dataset_name

        if samples is not None and class_to_idx is not None:
            self.samples = samples.copy()
            self.class_to_idx = class_to_idx
            if shuffle_samples:
                random.shuffle(self.samples)
            return

        dn = dataset_name.lower()
        if dn in ["german_csv", "csv"]:
            self.samples, self.class_to_idx = self._scan_csv(root, csv_filename)
        elif dn in ["belgium", "belgium_split", "belgium_1t1t", "belgium_1train_1test", "german_51k", "neu-det",
                    "neu-det_surface-dec"]:
            self.samples, self.class_to_idx = self._scan_belgium(root)
        else:
            self.samples, self.class_to_idx = self._scan_folder(root)

        if shuffle_samples:
            random.shuffle(self.samples)

    def _scan_folder(self, root):
        class_dirs = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))],
            key=natural_sort_key
        )
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}

        samples = []
        for class_name in class_dirs:
            class_path = os.path.join(root, class_name)
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(IMG_EXTS)]
            for img_file in image_files:
                samples.append((os.path.join(class_path, img_file), class_name))

        return samples, class_to_idx

    def _scan_csv(self, root, csv_filename):
        csv_path = os.path.join(root, csv_filename)
        samples = []
        classes_set = set()

        with open(csv_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = os.path.join(root, row['Path'])
                class_id = str(row['ClassId'])
                samples.append((img_path, class_id))
                classes_set.add(class_id)

        sorted_classes = sorted(list(classes_set), key=natural_sort_key)
        class_to_idx = {c: idx for idx, c in enumerate(sorted_classes)}

        return samples, class_to_idx

    def _scan_belgium(self, root):
        train_dir = self._find_subdir(root, ["Train", "train", "TRAIN"])
        test_dir = self._find_subdir(root, ["Test", "test", "TEST"])

        if not train_dir or not test_dir:
            raise FileNotFoundError(f"Không tìm thấy thư mục Train/Test trong: {root}")

        class_dirs = set()
        for sub_dir in [train_dir, test_dir]:
            class_dirs.update(d for d in os.listdir(sub_dir) if os.path.isdir(os.path.join(sub_dir, d)))
        class_dirs = sorted(class_dirs, key=natural_sort_key)
        class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}

        samples = []
        for sub_dir in [train_dir, test_dir]:
            for class_name in class_dirs:
                class_path = os.path.join(sub_dir, class_name)
                if not os.path.isdir(class_path):
                    continue
                image_files = [f for f in os.listdir(class_path) if f.lower().endswith(IMG_EXTS)]
                for img_file in image_files:
                    samples.append((os.path.join(class_path, img_file), class_name))

        return samples, class_to_idx

    @staticmethod
    def _find_subdir(root, candidates):
        for name in candidates:
            path = os.path.join(root, name)
            if os.path.isdir(path):
                return path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[class_name]
        return img, label


# =================================================================================
# TEST DATALOADER PREPARATION (EXACT 70-15-15 SPLIT)
# =================================================================================
def get_test_dataloader(root, dataset_name, batch_size, img_size=32, seed=42):
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    full_dataset = TrafficSignDataset(
        root=root,
        dataset_name=dataset_name,
        shuffle_samples=False
    )

    indices = list(range(len(full_dataset)))
    labels = [full_dataset.class_to_idx[full_dataset.samples[i][1]] for i in indices]

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, shuffle=True, stratify=labels
    )
    temp_labels = [labels[i] for i in temp_idx]

    _, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, shuffle=True, stratify=temp_labels
    )

    test_samples = [full_dataset.samples[i] for i in test_idx]

    test_dataset = TrafficSignDataset(
        root=root,
        dataset_name=dataset_name,
        transform=transform_test,
        samples=test_samples,
        class_to_idx=full_dataset.class_to_idx,
        shuffle_samples=False
    )

    num_workers = min(4, os.cpu_count() or 2)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader, len(full_dataset.class_to_idx)


# =================================================================================
# MODEL CHECKPOINT LOADER
# =================================================================================
def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ['model_state_dict', 'state_dict', 'model', 'net']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        clean_state_dict[name] = v

    model.load_state_dict(clean_state_dict, strict=False)
    return model


# =================================================================================
# EVALUATION & VISUALIZATION
# =================================================================================
def evaluate():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Device: {device}")
    print("Loading test data split (15% stratified test set)...")
    test_loader, num_classes = get_test_dataloader(
        root=DATASET_PATH,
        dataset_name=DATASET_NAME,
        batch_size=BATCH_SIZE,
        img_size=PICTURE_SIZE,
        seed=SEED
    )

    print(f"Initializing HybricMamba (Classes: {num_classes})...")
    model = Super_Mamba(dims=3, depth=4, num_classes=num_classes).to(device)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    print(f"Loading weights from: {CHECKPOINT_PATH}")
    model = load_checkpoint(model, CHECKPOINT_PATH, device)
    model.eval()

    all_targets = []
    all_predictions = []

    print("\nRunning inference on Test Set...")
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x = x.to(device, non_blocking=True).contiguous()
            y = y.to(device, non_blocking=True)

            preds = model(x)
            if isinstance(preds, tuple):
                preds = preds[0]

            predictions = preds.argmax(dim=1)
            all_targets.extend(y.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    all_targets = np.array(all_targets)
    all_predictions = np.array(all_predictions)

    # 1. Calculate Metrics
    accuracy = (all_predictions == all_targets).mean() * 100
    f1_macro = f1_score(all_targets, all_predictions, average='macro') * 100
    f1_weighted = f1_score(all_targets, all_predictions, average='weighted') * 100

    # 2. Format & Write Log Results
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_banner = "=" * 60
    log_text = (
        f"\n{log_banner}\n"
        f"TEST EVALUATION REPORT - {timestamp}\n"
        f"Checkpoint Path    : {CHECKPOINT_PATH}\n"
        f"Total Test Samples : {len(all_targets)}\n"
        f"Accuracy           : {accuracy:.2f}%\n"
        f"F1-Score (Macro)   : {f1_macro:.2f}%\n"
        f"F1-Score (Weighted): {f1_weighted:.2f}%\n"
        f"{log_banner}"
    )

    log_and_print(log_text, LOG_FILE_PATH)

    # 3. Confusion Matrix Plot
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    fig_size = max(10, int(num_classes * 0.4))

    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {DATASET_NAME} (Acc: {accuracy:.2f}%)", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Confusion Matrix heatmap saved to: {cm_path}")
    print(f"✓ Evaluation log appended to: {LOG_FILE_PATH}")


if __name__ == "__main__":
    evaluate()