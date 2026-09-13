# coding=utf-8
# @FileName: train_clean.py
# @Author: CZH (cleaned)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from tqdm import tqdm
import argparse
import numpy as np
import random
import logging
import torchvision.models as models
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from models. import Super_Mamba


# ─────────────────────────────────────────────
# parameters
# ─────────────────────────────────────────────

model_name = "SupperMamba_convet_3_6"
save_path = "/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult"
seed = 42
picture_size=32
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',      default="German",  type=str)
    parser.add_argument('--class_num',         default=43,       type=int)
    parser.add_argument('--batch_size',        default=64,        type=int)
    parser.add_argument('--start_epoch',       default=0,         type=int)
    parser.add_argument('--num_epoch',         default=50,       type=int)
    parser.add_argument('--root_dataset_path',
                        default="/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/dataset",
                        type=str)
    return parser.parse_args()

# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class TrafficSignFullDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []

        for domain in os.listdir(root):
            domain_path = os.path.join(root, domain)

            if not os.path.isdir(domain_path):
                continue
            for class_name in os.listdir(domain_path):
                class_path = os.path.join(domain_path, class_name)
                if not os.path.isdir(class_path):
                    continue
                for img_name in os.listdir(class_path):
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


# ─────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────

def dataloader_prepare(root, batchsize, test_ratio=0.2):
    transform_train = transforms.Compose([
        transforms.Resize((
picture_size,
picture_size)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((
picture_size,
picture_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    full_dataset = TrafficSignFullDataset(root)
    num_classes = len(full_dataset.class_to_idx)
    print(f"Số nhãn (classes): {num_classes}")
    print(f"Danh sách nhãn: {list(full_dataset.class_to_idx.keys())}")




    indices = list(range(len(full_dataset)))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=42,
        shuffle=True,
        stratify=[full_dataset.samples[i][1] for i in indices]
    )

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    test_dataset  = torch.utils.data.Subset(full_dataset, test_idx)


    train_dataset.dataset = TrafficSignFullDataset(root, transform=transform_train)
    test_dataset.dataset  = TrafficSignFullDataset(root, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batchsize,
                              shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batchsize,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")
    return train_loader, test_loader


# ─────────────────────────────────────────────
# Train & Test
# ─────────────────────────────────────────────

def train_and_test(start_epoch, num_epochs,
                   train_dataloader, test_dataloader,
                   model, device, batchsize, class_num,
                   model_name, dataset_name, path):

    # Tạo thư mục lưu kết quả
    folder_path = os.path.join(path, model_name, dataset_name)
    os.makedirs(folder_path, exist_ok=True)

    # Logging
    logging.basicConfig(
        filename=os.path.join(folder_path, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    logging.info(f"Start Training — model: {model_name}")

    save_path = os.path.join(folder_path, f"{model_name}_train_on_{dataset_name}.pth")

    # Optimizer / Scheduler / Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                                momentum=0.9, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 40], gamma=0.1
    )

    model = model.to(device)
    best_acc = 0.0
    best_acc_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        # ── Train ──
        model.train()
        running_loss = 0.0

        with tqdm(train_dataloader, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                tepoch.set_postfix(loss=f"{running_loss / batchsize:.4f}")

        scheduler.step()

        # ── Evaluate ──
        model.eval()
        num_correct = num_samples = 0

        with torch.no_grad():
            for x, y in test_dataloader:
                x, y = x.to(device), y.to(device)
                preds = model(x)
                predictions = preds.argmax(dim=1)
                num_correct += (predictions == y).sum().item()
                num_samples += y.size(0)

        acc = num_correct / num_samples
        lr  = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}  Accuracy: {acc:.4f}  LR: {lr}")
        logging.info(f"Epoch {epoch}/{num_epochs}  Accuracy: {acc:.4f}  "
                     f"Loss: {running_loss / batchsize:.4f}")

        if acc > best_acc:
            best_acc       = acc
            best_acc_epoch = epoch
            torch.save({
                'epoch':              epoch,
                'model_state_dict':   model.state_dict(),
                'optim_state_dict':   optimizer.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
            }, save_path)

        logging.info(f"Best Accuracy: {best_acc:.4f} at epoch {best_acc_epoch}")

    # ── Confusion Matrix (top 15 classes) ──
    model.eval()
    true_labels, predicted_labels = [], []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(30, 28))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    cm_path = os.path.join(folder_path, "confusion_matrix_all.png")
    plt.savefig(cm_path, dpi=300)
    plt.show()
    print(f"Confusion matrix saved → {cm_path}")


# ─────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────



def get_fps(model):
    # 定义图像转换
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),  # 调整图像大小
    #     transforms.ToTensor(),  # 转换为张量
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    # ])
    #
    # # 加载图像
    # image_path = r'F:\acm\pythonProject\process_dataset\China\test\prohibitory\3.5m\271.jpg'  # 替换为你自己的图像路径
    # image = Image.open(image_path)
    #
    # # 图像转换和预处理
    # input_tensor = transform(image)
    # input_batch = input_tensor.unsqueeze(0)  # 添加批次维度

    iterations = 300  # 重复计算的轮次

    model = model
    device = torch.device("cuda:0")
    # model.to(device)

    random_input = torch.randn(1, 3, 224, 224).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # GPU预热
    for _ in range(50):
        _ = model(random_input)

    # 测速
    times = torch.zeros(iterations)  # 存储每轮iteration的时间
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(random_input)
            ender.record()
            # 同步GPU时间
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 计算时间
            times[iter] = curr_time
            # print(curr_time)

    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time * 1.0, 1000.0 / mean_time))
    inference_time = mean_time
    fps = 1000.0 / mean_time
    return inference_time, fps

# ─────────────────────────────────────────────
# models
# ─────────────────────────────────────────────
class ModifiedVGG16(nn.Module):
    def __init__(self, num_classes=43):
        super(ModifiedVGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=False)
        num_features = self.vgg16.classifier[-1].in_features
        self.vgg16.classifier[-1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vgg16(x)
# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    # Reproducibility

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args   = get_args()

    train_dataloader, test_dataloader = dataloader_prepare(
        args.root_dataset_path, args.batch_size
    )
    model = Super_Mamba(dims=3, depth=6, num_classes=args.class_num)
    train_and_test(
        args.start_epoch, args.num_epoch,
        train_dataloader, test_dataloader,
        model, device, args.batch_size,
        args.class_num, model_name,
        args.dataset_name, path=save_path
    )
    get_fps(model)