import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.CNN_Mamba_CNN_Mamba_Enhanced.HybricMamba import HybricMamba
from models.vmamba.Vmamba_ultils import Super_Mamba

# Cấu hình đường dẫn
DATASET_PATH = "/home/biu-linux/DeepLearning_Projects/DoAnNganh/dataset_reOrgan"
CHECKPOINT_HYBRIC_PATH = "/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult/TFJ/Heavy_HybricMamba/German/Heavy_HybricMamba_best.pth"
CHECKPOINT_SUPER_PATH = "/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult/TFJ/Super_Mamba_dim_4/German/Super_Mamba_dim_4_best.pth"
OUTPUT_TXT_PATH = "Full_imPth_both_wrong_image.txt"


def create_hybric_mamba():
    return HybricMamba(
        dims=(3, 32, 64, 112, 176),
        num_classes=43,
        mbconv_expand_ratio=6,
        ssm_d_state=16,
        ssm_ratio=2.0,
        mamba_blocks=(2, 3),
        cnn_blocks=(2, 2),
        ssm_frac=0.7,
        conv_frac=0.2,
        use_aux=True,
    )


def create_super_mamba():
    return Super_Mamba(
        dims=3,
        depth=4,
        num_classes=43
    )


def load_model(model_instance, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    else:
        state_dict = checkpoint

    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()} if isinstance(state_dict, dict) else state_dict
    model_instance.load_state_dict(clean_state_dict, strict=False)
    model_instance.to(device)
    model_instance.eval()
    return model_instance


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("📦 Đang đọc dataset...")
    dataset = datasets.ImageFolder(root=DATASET_PATH, transform=transform)

    classes = sorted(dataset.classes, key=lambda x: int(x))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    dataset.classes = classes
    dataset.class_to_idx = class_to_idx
    dataset.samples = dataset.make_dataset(DATASET_PATH, class_to_idx, dataset.extensions)
    dataset.targets = [s[1] for s in dataset.samples]

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    print("📦 Đang nạp HybricMamba...")
    model_hybric = load_model(create_hybric_mamba(), CHECKPOINT_HYBRIC_PATH, device)

    print("📦 Đang nạp Super_Mamba...")
    model_super = load_model(create_super_mamba(), CHECKPOINT_SUPER_PATH, device)

    print("✅ Load cả 2 models thành công!")

    both_wrong_list = []
    total_images = len(dataset)
    global_idx = 0

    print(f"\n🔍 Đang chạy so sánh dự đoán trên toàn bộ dataset ({total_images:,} ảnh)...")

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Processing"):
            images = images.to(device)

            # Dự đoán từ HybricMamba
            outputs_hybric = model_hybric(images)
            if isinstance(outputs_hybric, tuple):
                outputs_hybric = outputs_hybric[0]
            probs_hybric = torch.softmax(outputs_hybric, dim=1)
            preds_hybric = torch.argmax(probs_hybric, dim=1)

            # Dự đoán từ Super_Mamba
            outputs_super = model_super(images)
            if isinstance(outputs_super, tuple):
                outputs_super = outputs_super[0]
            probs_super = torch.softmax(outputs_super, dim=1)
            preds_super = torch.argmax(probs_super, dim=1)

            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_h = preds_hybric[i].item()
                pred_s = preds_super[i].item()

                # Điều kiện: CẢ HAI model đều đoán SAI so với nhãn thực tế
                if pred_h != true_label and pred_s != true_label:
                    conf_h = probs_hybric[i][pred_h].item() * 100
                    conf_s = probs_super[i][pred_s].item() * 100
                    img_path = dataset.samples[global_idx + i][0]

                    both_wrong_list.append((img_path, true_label, pred_h, conf_h, pred_s, conf_s))

            global_idx += len(labels)

    # Ghi danh sách ra file text
    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 110 + "\n")
        f.write("DANH SÁCH ẢNH BỊ DỰ ĐOÁN SAI BỞI CẢ 2 MÔ HÌNH (HybricMamba & Super_Mamba)\n")
        f.write(f"Tổng số ảnh trong dataset: {total_images:,} ảnh\n")
        f.write(f"Số lượng ảnh cả 2 mô hình cùng đoán sai: {len(both_wrong_list):,} ảnh\n")
        if total_images > 0:
            error_rate = (len(both_wrong_list) / total_images) * 100
            f.write(f"Tỷ lệ lỗi chung (Both Wrong Rate): {error_rate:.2f}%\n")
        f.write("=" * 110 + "\n\n")

        for path, true_cls, pred_h, conf_h, pred_s, conf_s in both_wrong_list:
            f.write(f"[True: {true_cls:02d} | Hybric: {pred_h:02d} ({conf_h:5.1f}%) | Super: {pred_s:02d} ({conf_s:5.1f}%)] Path: {path}\n")

    print("\n" + "=" * 50)
    print(f"✅ Hoàn thành!")
    print(f"📊 Đã quét {total_images:,} ảnh thuộc dataset.")
    print(f"❌ Phát hiện {len(both_wrong_list):,} ảnh bị CẢ HAI mô hình đoán sai.")
    print(f"📄 Kết quả chi tiết đã lưu tại: {OUTPUT_TXT_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()