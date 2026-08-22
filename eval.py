import os
import random
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from models.CNN_Mamba_CNN_Mamba_Enhanced.HybricMamba import HybricMamba

SEED = 42
PICTURE_SIZE = 32
BATCH_SIZE = 64
DATASET_PATH = "/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/dataset"
CHECKPOINT_PATH_9716 = "/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult/TFJ/CNNMambaCNNMambaEnhancedV4/German/CNNMambaCNNMambaEnhancedV4_train_on_German.pth"


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrafficSignFullDataset(Dataset):
    def __init__(self, root, transform=None, samples=None, class_to_idx=None):
        self.root = root
        self.transform = transform

        if samples is not None:
            self.samples = samples
            self.class_to_idx = class_to_idx
            return

        self.samples = []
        for domain in sorted(os.listdir(root)):
            domain_path = os.path.join(root, domain)
            if not os.path.isdir(domain_path):
                continue

            for class_name in sorted(os.listdir(domain_path)):
                class_path = os.path.join(domain_path, class_name)
                if not os.path.isdir(class_path):
                    continue

                for img_name in sorted(os.listdir(class_path)):
                    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        self.samples.append((img_path, class_name))

        classes = sorted(set(s[1] for s in self.samples))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[class_name]
        return img, label


def get_test_loader(root, batch_size, test_ratio=0.2):
    transform_test = transforms.Compose([
        transforms.Resize((PICTURE_SIZE, PICTURE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    full_dataset = TrafficSignFullDataset(root)
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.samples[i][1] for i in indices]

    _, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=SEED,
        shuffle=True,
        stratify=labels
    )

    test_samples = [full_dataset.samples[i] for i in test_idx]

    test_dataset = TrafficSignFullDataset(
        root=root,
        transform=transform_test,
        samples=test_samples,
        class_to_idx=full_dataset.class_to_idx
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return test_loader


def evaluate():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị sử dụng: {device}")

    # 1. Chuẩn bị DataLoader
    print("Đang tải dữ liệu...")
    test_loader = get_test_loader(DATASET_PATH, BATCH_SIZE)

    # 2. Khởi tạo Mô hình
    print("Đang khởi tạo mô hình CNNMambaCNNMambaEnhancedV4...")
    model = HybricMamba(
        dims=(3, 16, 32, 56, 96),
        num_classes=43,
        mbconv_expand_ratio=4,
        ssm_d_state=8,
        mamba_blocks=(1, 1),
        ssm_frac=0.5,
        conv_frac=0.3,
        use_aux=True,
    ).to(device)

    # 3. Tải Checkpoint
    if not os.path.exists(CHECKPOINT_PATH_9716):
        raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {CHECKPOINT_PATH_9716}")

    print(f"Đang load weights từ file: {CHECKPOINT_PATH_9716}")
    checkpoint = torch.load(CHECKPOINT_PATH_9716, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        saved_epoch = checkpoint.get('epoch', 'N/A')
        saved_acc = checkpoint.get('best_acc', None)
        print(f"-> Load thành công checkpoint từ Epoch {saved_epoch}")
        if saved_acc is not None:
            print(f"-> Accuracy ghi nhận lúc lưu: {saved_acc * 100:.2f}%")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    num_correct = 0
    num_samples = 0

    print("\nBắt đầu kiểm thử...")
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            preds = model(x)
            if isinstance(preds, tuple):
                preds = preds[0]

            predictions = preds.argmax(dim=1)
            num_correct += (predictions == y).sum().item()
            num_samples += y.size(0)

    accuracy = (num_correct / num_samples) * 100
    print("\n" + "=" * 50)
    print(f"Tổng số ảnh kiểm thử: {num_samples}")
    print(f"Số ảnh dự đoán đúng : {num_correct}")
    print(f"Độ chính xác (Accuracy): {accuracy:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    evaluate()