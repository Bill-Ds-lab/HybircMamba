

import argparse
import csv
import gc
import os
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from Dataloader.DATASET import TrafficSignDataset
from data_split_utils import get_or_create_split
from models.CNN_Mamba_CNN_Mamba_Enhanced.HybricMamba import HybricMamba
from models.vmamba.Vmamba_ultils import Super_Mamba


def get_args():
    parser = argparse.ArgumentParser(description="Benchmark & So sánh các mô hình Traffic Sign")

    parser.add_argument('--dataset_name', default="German_51k", type=str,
                         choices=["German", "German_CSV", "Belgium", "German_51k", "NEU-DET_surface-dec"])
    parser.add_argument('--csv_filename', default="Train.csv", type=str)
    parser.add_argument('--root_dataset_path',
                         default="/home/biu-linux/DeepLearning_Projects/DoAnNganh/data/German_51k", type=str)
    parser.add_argument('--save_path',
                         default="/home/biu-linux/DeepLearning_Projects/DoAnNganh/HybricMamba/Ressult/TFJ", type=str)
    parser.add_argument('--output_dir', default="./benchmark_outputs", type=str)

    parser.add_argument('--picture_size', default=32, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--SEED', default=2223, type=int)

    parser.add_argument('--models', nargs='+', default=[
        "LIGHT_HYBRIC_MAMBA",
        "MEDIUM_HYBRIC_MAMBA",
        "HEAVY_HYBRIC_MAMBA",
        "SUPER_MAMBA_DEPT_3",
        "SUPER_MAMBA_DEPT_4",
        "VGG16",
        "RESNET18",
        "VIT_B",
        "VIT_S",
        "EFFICIENTNET_B0",
        "MOBILENETV3_SMALL",
        "GHOSTNET",
    ])

    parser.add_argument('--skip_missing_checkpoint', action='store_true', default=True)
    parser.add_argument('--latency_batch_sizes', nargs='+', type=int, default=[1, 8, 16, 32])
    parser.add_argument('--n_warmup', default=30, type=int)
    parser.add_argument('--n_runs', default=200, type=int)
    parser.add_argument('--n_latency_repeats', default=5, type=int)

    return parser.parse_args()


def auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _friendly_cuda_error(e: Exception, device: str) -> str:
    msg = str(e)
    if "is_cuda" in msg or "CUDA" in msg:
        return f"Mamba selective-scan kernel yêu cầu GPU (device='{device}'). Lỗi gốc: {msg}"
    return msg


# --------------------------------------------------------------------------- #
# BUILD MODEL - ĐÃ SỬA ĐÚNG 100% KHỚP VỚI train.py
# --------------------------------------------------------------------------- #
def build_Model(name, num_classes=43, pretrained=False):
    if name == "LIGHT_HYBRIC_MAMBA":
        return HybricMamba(
            dims=(3, 16, 32, 56, 96),
            num_classes=num_classes,
            mbconv_expand_ratio=4,
            ssm_d_state=8,
            mamba_blocks=(1, 1),
            ssm_frac=0.5,
            conv_frac=0.3,
            use_aux=True,
        )
    elif name == "MEDIUM_HYBRIC_MAMBA":
        return HybricMamba(
            dims=(3, 24, 48, 80, 128),
            num_classes=num_classes,
            mbconv_expand_ratio=4,
            ssm_d_state=12,
            ssm_ratio=1.5,
            mamba_blocks=(2, 2),
            cnn_blocks=(1, 2),
            ssm_frac=0.6,
            conv_frac=0.25,
            use_aux=True,
        )
    elif name == "HEAVY_HYBRIC_MAMBA":
        return HybricMamba(
            dims=(3, 32, 64, 112, 176),
            num_classes=num_classes,
            mbconv_expand_ratio=6,
            ssm_d_state=16,
            ssm_ratio=2.0,
            mamba_blocks=(2, 3),
            cnn_blocks=(2, 2),
            ssm_frac=0.7,
            conv_frac=0.2,
            use_aux=True,
        )
    elif name == "SUPER_MAMBA_DEPT_4":
        return Super_Mamba(dims=3, depth=4, num_classes=num_classes)
    elif name == "SUPER_MAMBA_DEPT_3":
        return Super_Mamba(dims=3, depth=3, num_classes=num_classes)
    elif name in ["VGG16", "VGG-16"]:
        return timm.create_model("vgg16", pretrained=pretrained, num_classes=num_classes)
    elif name in ["RESNET18", "ResNet18"]:
        return timm.create_model("resnet18", pretrained=pretrained, num_classes=num_classes)
    elif name in ["VIT_B", "ViT-B"]:
        return timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes, img_size=32)
    elif name in ["VIT_S", "ViT-S"]:
        return timm.create_model("vit_small_patch16_224", pretrained=pretrained, num_classes=num_classes, img_size=32)
    elif name in ["EFFICIENTNET_B0", "EfficientNet-B0"]:
        return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=num_classes)
    elif name in ["MOBILENETV3_SMALL", "MobileNetV3-Small"]:
        return timm.create_model("mobilenetv3_small_100", pretrained=pretrained, num_classes=num_classes)
    elif name in ["GHOSTNET", "GhostNet"]:
        return timm.create_model("ghostnet_100", pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Tên mô hình '{name}' không hợp lệ.")


def load_checkpoint_safely(model, checkpoint_path, device):
    print(f"[LOAD] Nạp checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "net"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    clean_state_dict = {
        (k.replace("module.", "") if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    missing_keys, unexpected_keys = model.load_state_dict(clean_state_dict, strict=False)

    if missing_keys:
        print(f"  ⚠️ Thiếu {len(missing_keys)} keys (VD: {missing_keys[:3]})")
    if unexpected_keys:
        print(f"  ⚠️ Thừa {len(unexpected_keys)} keys (VD: {unexpected_keys[:3]})")
    if not missing_keys and not unexpected_keys:
        print("  ✅ Checkpoint khớp 100%.")

    best_val_acc = checkpoint.get("best_val_acc", None) if isinstance(checkpoint, dict) else None

    del checkpoint, state_dict, clean_state_dict
    return model, missing_keys, unexpected_keys, best_val_acc


def find_checkpoint(save_path, model_name, dataset_name):
    path = os.path.join(save_path, model_name, dataset_name, f"{model_name}_best.pth")
    return path if os.path.exists(path) else None


def get_transforms(img_size=32):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


# --------------------------------------------------------------------------- #
# TEST LOADER - DÙNG SPLIT CỐ ĐỊNH THEO PATH (miễn nhiễm với shuffle order)
# --------------------------------------------------------------------------- #
def build_test_loader(args):
    # shuffle_samples không còn quan trọng nữa vì get_or_create_split match theo path,
    # nhưng để nhất quán và tránh nhầm lẫn, ta để False ở đây.
    full_dataset = TrafficSignDataset(
        root=args.root_dataset_path,
        dataset_name=args.dataset_name,
        csv_filename=args.csv_filename,
        shuffle_samples=False,
    )
    num_classes = len(full_dataset.class_to_idx)
    dataset_class = type(full_dataset)

    split_path_exists = os.path.exists(
        os.path.join(args.save_path, "_dataset_splits", f"{args.dataset_name}_split_seed{args.SEED}.json")
    )
    if not split_path_exists:
        print("\n" + "!" * 90)
        print("⚠️  CẢNH BÁO QUAN TRỌNG: Chưa có file split cố định cho dataset này.")
        print("   Split sẽ được TẠO MỚI ngay bây giờ và lưu lại cho các lần sau.")
        print("   NHƯNG: nếu các checkpoint *_best.pth hiện có được train TRƯỚC KHI")
        print("   bạn áp dụng cơ chế split cố định này cho train.py, thì split mới")
        print("   tạo ở đây CÓ THỂ KHÔNG khớp với phần dữ liệu model đã học lúc train.")
        print("   -> Để kết quả benchmark đáng tin cậy, hãy: ")
        print("      1) Sửa train.py để dùng chung data_split_utils.get_or_create_split()")
        print("      2) Train lại (ít nhất 1 lần) TRƯỚC KHI tin vào accuracy ở đây.")
        print("!" * 90 + "\n")

    _, _, test_samples = get_or_create_split(
        full_dataset, args.save_path, args.dataset_name, seed=args.SEED
    )

    transform_test = get_transforms(args.picture_size)
    test_dataset = dataset_class(
        root=args.root_dataset_path,
        transform=transform_test,
        samples=test_samples,
        class_to_idx=full_dataset.class_to_idx,
        shuffle_samples=False,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    print(f"[DATA] Test set: {len(test_dataset)} ảnh | {num_classes} classes")
    return test_loader, num_classes


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 ** 2)
    return {"total_params": total, "trainable_params": trainable, "size_mb": size_mb, "size_m": total / 1e6}


def count_flops(model: nn.Module, input_size=(1, 3, 32, 32), device="cpu") -> dict:
    model = model.to(device).eval()
    dummy_input = torch.randn(*input_size).to(device)
    try:
        from thop import profile
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        method = "thop"
    except Exception as e1:
        try:
            from fvcore.nn import FlopCountAnalysis
            fca = FlopCountAnalysis(model, dummy_input)
            flops = fca.total()
            method = "fvcore"
        except Exception as e2:
            raise RuntimeError(f"Không thể tính FLOPs (thop: {e1}; fvcore: {e2})")
    return {"flops_m": flops / 1e6, "method": method}


def measure_inference_time(model, input_size=(1, 3, 32, 32), device="cpu", n_warmup=30, n_runs=200) -> dict:
    model = model.to(device).eval()
    dummy_input = torch.randn(*input_size).to(device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    with torch.no_grad():
        for _ in range(n_warmup):
            try:
                _ = model(dummy_input)
            except RuntimeError as e:
                raise RuntimeError(_friendly_cuda_error(e, device)) from e

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        if device == "cuda":
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(n_runs):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))
        else:
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(dummy_input)
                times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    return {"median_ms": float(np.median(times))}


def measure_peak_memory(model, input_size=(1, 3, 32, 32), device="cuda") -> dict:
    if device != "cuda" or not torch.cuda.is_available():
        return {"peak_memory_mb": None}
    model = model.to(device).eval()
    dummy_input = torch.randn(*input_size).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return {"peak_memory_mb": peak_mb}


def evaluate_accuracy(model, dataloader, device="cpu"):
    model = model.to(device).eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

            label_arr = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            all_labels.extend(label_arr)

    acc = accuracy_score(all_labels, all_preds) * 100
    f1_macro = f1_score(all_labels, all_preds, average="macro") * 100
    f1_weighted = f1_score(all_labels, all_preds, average="weighted") * 100
    cm = confusion_matrix(all_labels, all_preds)
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted, "confusion_matrix": cm}


def plot_confusion_matrix(cm, model_name, save_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def compute_ids(accuracy: float, params_million: float) -> float:
    return accuracy / params_million if params_million > 0 else 0.0


def benchmark_one_model(model_name, args, test_loader, num_classes, device):
    print(f"\n{'=' * 70}\n>>> BENCHMARK: {model_name}\n{'=' * 70}")
    result = {"model_name": model_name}

    model = build_Model(model_name, num_classes=num_classes, pretrained=False)
    params_info = count_parameters(model)
    result["params_m"] = params_info["size_m"]
    result["size_mb"] = params_info["size_mb"]

    ckpt_path = find_checkpoint(args.save_path, model_name, args.dataset_name)
    result["checkpoint_found"] = ckpt_path is not None

    if ckpt_path is not None:
        model, missing, unexpected, _ = load_checkpoint_safely(model, ckpt_path, device)
        result["arch_mismatch"] = bool(missing or unexpected)
    else:
        print(
            f"  ⚠️ Không tìm thấy checkpoint tại: {args.save_path}/{model_name}/{args.dataset_name}/{model_name}_best.pth")
        result["arch_mismatch"] = None

    model = model.to(device)

    try:
        flops_info = count_flops(model, input_size=(1, 3, args.picture_size, args.picture_size), device=device)
        result["flops_m"] = flops_info["flops_m"]
    except Exception as e:
        print(f"  FLOPs lỗi: {_friendly_cuda_error(e, device)}")
        result["flops_m"] = None

    result["latency_ms"] = {}
    for bs in args.latency_batch_sizes:
        try:
            medians = []
            for _ in range(args.n_latency_repeats):
                t_info = measure_inference_time(
                    model, input_size=(bs, 3, args.picture_size, args.picture_size),
                    device=device, n_warmup=args.n_warmup, n_runs=args.n_runs,
                )
                medians.append(t_info["median_ms"])
                time.sleep(0.05)
            result["latency_ms"][bs] = float(np.median(medians))
            print(f"  Latency (bs={bs}): {result['latency_ms'][bs]:.3f} ms")
        except RuntimeError as e:
            result["latency_ms"][bs] = None
            print(f"  Latency (bs={bs}) LỖI: {e}")

    try:
        mem_info = measure_peak_memory(model, input_size=(1, 3, args.picture_size, args.picture_size), device=device)
        result["peak_memory_mb"] = mem_info["peak_memory_mb"]
    except Exception as e:
        result["peak_memory_mb"] = None
        print(f"  Memory lỗi: {e}")

    if ckpt_path is not None and not result["arch_mismatch"]:
        try:
            acc_info = evaluate_accuracy(model, test_loader, device=device)
            result["accuracy"] = acc_info["accuracy"]
            result["f1_macro"] = acc_info["f1_macro"]
            result["f1_weighted"] = acc_info["f1_weighted"]
            result["ids"] = compute_ids(acc_info["accuracy"], params_info["size_m"])

            cm_path = plot_confusion_matrix(acc_info["confusion_matrix"], model_name, args.output_dir)
            result["confusion_matrix_path"] = cm_path
            print(
                f"  ✅ Test Acc: {acc_info['accuracy']:.2f}% | F1-macro: {acc_info['f1_macro']:.2f}% | IDS: {result['ids']:.2f}")
        except Exception as e:
            print(f"  ❌ Lỗi khi tính accuracy: {e}")
            result["accuracy"] = None
    elif result["arch_mismatch"]:
        print("  ⚠️ Bỏ qua accuracy vì kiến trúc không khớp checkpoint.")
        result["accuracy"] = None
    else:
        result["accuracy"] = None

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return result


def main():
    args = get_args()
    device = auto_device()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print(f"SO SÁNH MÔ HÌNH | Dataset: {args.dataset_name} | Device: {device}")
    print("=" * 80)

    test_loader, num_classes = build_test_loader(args)

    all_results = []
    for model_name in args.models:
        try:
            res = benchmark_one_model(model_name, args, test_loader, num_classes, device)
        except Exception as e:
            print(f"❌ LỖI với model {model_name}: {e}")
            res = {"model_name": model_name, "error": str(e)}
        all_results.append(res)

    print("\n" + "=" * 110)
    print("BẢNG TỔNG HỢP KẾT QUẢ")
    print("=" * 110)
    header = f"{'Model':<22s} | {'Params(M)':<10s} | {'FLOPs(M)':<10s} | {'Lat bs=1(ms)':<12s} | {'Mem(MB)':<9s} | {'Acc(%)':<8s} | {'F1-macro':<9s} | {'IDS':<8s}"
    print(header)
    print("-" * 110)

    csv_rows = []
    for r in all_results:
        params_m = f"{r.get('params_m', 0):.3f}" if r.get("params_m") is not None else "N/A"
        flops_m = f"{r.get('flops_m', 0):.2f}" if r.get("flops_m") is not None else "N/A"
        lat1 = r.get("latency_ms", {}).get(1) if r.get("latency_ms") else None
        lat1_str = f"{lat1:.3f}" if lat1 is not None else "N/A"
        mem = f"{r.get('peak_memory_mb'):.1f}" if r.get("peak_memory_mb") is not None else "N/A"
        acc = f"{r.get('accuracy'):.2f}" if r.get("accuracy") is not None else "N/A"
        f1m = f"{r.get('f1_macro'):.2f}" if r.get("f1_macro") is not None else "N/A"
        ids = f"{r.get('ids'):.2f}" if r.get("ids") is not None else "N/A"

        print(
            f"{r['model_name']:<22s} | {params_m:<10s} | {flops_m:<10s} | {lat1_str:<12s} | {mem:<9s} | {acc:<8s} | {f1m:<9s} | {ids:<8s}")

        csv_rows.append({
            "model_name": r["model_name"],
            "params_m": r.get("params_m"),
            "flops_m": r.get("flops_m"),
            **{f"latency_bs{bs}_ms": r.get("latency_ms", {}).get(bs) for bs in args.latency_batch_sizes},
            "peak_memory_mb": r.get("peak_memory_mb"),
            "accuracy": r.get("accuracy"),
            "f1_macro": r.get("f1_macro"),
            "f1_weighted": r.get("f1_weighted"),
            "ids": r.get("ids"),
            "checkpoint_found": r.get("checkpoint_found"),
            "arch_mismatch": r.get("arch_mismatch"),
        })

    print("=" * 110)

    csv_path = os.path.join(args.output_dir, f"benchmark_results_{args.dataset_name}.csv")
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n📄 Đã lưu kết quả CSV: {csv_path}")


if __name__ == "__main__":
    main()