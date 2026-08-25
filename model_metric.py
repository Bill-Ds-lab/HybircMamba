import gc
import time
import numpy as np
import timm
import torch
import torch.nn as nn

# Giữ nguyên các import mô hình của bạn
from models.CNN_Mamba_CNN_Mamba_Enhanced.HybricMamba import HybricMamba
from models.vmamba.Vmamba_ultils import Super_Mamba


def auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _friendly_cuda_error(e: Exception, device: str) -> str:
    msg = str(e)
    if "is_cuda" in msg or "CUDA" in msg:
        return (
            f"Model này dùng Mamba selective-scan CUDA kernel nên chỉ chạy "
            f"được trên GPU, không chạy được trên CPU (device='{device}'). "
            f"Hãy gọi lại với device='cuda' (cần máy có GPU + driver CUDA "
            f"phù hợp với bản mamba_ssm đã cài). Lỗi gốc: {msg}"
        )
    return msg


# --------------------------------------------------------------------------- #
# 1. SỐ LƯỢNG THAM SỐ
# --------------------------------------------------------------------------- #
def count_parameters(model: nn.Module) -> dict:
    """Trả về tổng số tham số, số tham số trainable, và kích thước ước tính (MB, float32)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024**2)  # float32 = 4 bytes

    return {
        "total_params": total,
        "trainable_params": trainable,
        "size_mb": size_mb,
        "size_m": total / 1e6,
    }


def count_by_top_module(model: nn.Module) -> dict:
    """Phân bố số tham số theo từng module con (top-level)."""
    breakdown = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        breakdown[name] = n
    return breakdown


# --------------------------------------------------------------------------- #
# 2. FLOPs
# --------------------------------------------------------------------------- #
def count_flops(
    model: nn.Module, input_size=(1, 3, 32, 32), device="cpu"
) -> dict:
    """Tính FLOPs của model bằng `thop` hoặc `fvcore`."""
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
            params = sum(p.numel() for p in model.parameters())
            method = "fvcore"
        except Exception as e2:
            raise RuntimeError(
                f"Không thể tính FLOPs. Lỗi thop: {e1}; Lỗi fvcore: {e2}."
            )

    return {
        "flops": flops,
        "flops_k": flops / 1e3,
        "flops_m": flops / 1e6,
        "flops_g": flops / 1e9,
        "params_from_flops_tool": params,
        "method": method,
    }


# --------------------------------------------------------------------------- #
# 3. THỜI GIAN SUY LUẬN (INFERENCE TIME)
# --------------------------------------------------------------------------- #
def measure_inference_time(
    model: nn.Module,
    input_size=(1, 3, 32, 32),
    device="cpu",
    n_warmup=100,
    n_runs=800,
) -> dict:
    """Đo thời gian suy luận trung bình cho 1 batch."""
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
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(n_runs):
                starter.record()
                _ = model(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender))  # ms
        else:
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = model(dummy_input)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # ms

    times = np.array(times)
    return {
        "mean_ms": float(times.mean()),
        "std_ms": float(times.std()),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "median_ms": float(np.median(times)),
        "n_runs": n_runs,
        "device": device,
    }


# --------------------------------------------------------------------------- #
# 4. ACCURACY + CONFUSION MATRIX
# --------------------------------------------------------------------------- #
def evaluate_accuracy(
    model: nn.Module, dataloader, device="cpu", class_names=None
):
    from sklearn.metrics import accuracy_score, confusion_matrix

    model = model.to(device).eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "y_true": all_labels,
        "y_pred": all_preds,
        "class_names": class_names,
    }


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


# --------------------------------------------------------------------------- #
# 5. INFORMATION DENSITY (IDS)
# --------------------------------------------------------------------------- #
def compute_ids(accuracy: float, params_million: float) -> float:
    return accuracy / params_million if params_million > 0 else 0.0


# --------------------------------------------------------------------------- #
# 6. ROBUSTNESS TEST
# --------------------------------------------------------------------------- #
def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 if gamma == 0 else gamma
    img = np.clip(image, 0, 255).astype(np.float32)
    out = np.power(img / 255.0, inv) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def add_gaussian_noise(
    image: np.ndarray, mean: float = 0, std: float = 25
) -> np.ndarray:
    img = image.astype(np.float32)
    noise = np.random.normal(mean, std, img.shape)
    out = img + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_rain_effect(
    image: np.ndarray,
    size_factor: float = 3.0,
    angle_range=(-60, 60),
    density: float = 0.02,
) -> np.ndarray:
    import cv2

    img = image.copy()
    h, w = img.shape[:2]
    n_drops = int(h * w * density)

    for _ in range(n_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        length = int(size_factor * np.random.uniform(2, 5))
        angle = np.deg2rad(np.random.uniform(*angle_range))
        x2 = int(x + length * np.sin(angle))
        y2 = int(y + length * np.cos(angle))
        cv2.line(img, (x, y), (x2, y2), (255, 255, 255), 1)

    return img


def robustness_test(
    model: nn.Module,
    dataloader_factory,
    device="cpu",
    scenarios: dict = None,
) -> dict:
    if scenarios is None:
        scenarios = {
            "dim_gamma_0.5": lambda img: adjust_gamma(img, 0.5),
            "dim_gamma_0.2": lambda img: adjust_gamma(img, 0.2),
            "exposure_gamma_1.5": lambda img: adjust_gamma(img, 1.5),
            "exposure_gamma_1.8": lambda img: adjust_gamma(img, 1.8),
            "rain_s3": lambda img: add_rain_effect(img, size_factor=3.0),
            "rain_s7": lambda img: add_rain_effect(img, size_factor=7.0),
            "noise_mean0": lambda img: add_gaussian_noise(
                img, mean=0, std=25
            ),
            "noise_mean-120": lambda img: add_gaussian_noise(
                img, mean=-120, std=25
            ),
            "noise_mean120": lambda img: add_gaussian_noise(
                img, mean=120, std=25
            ),
        }

    results = {}
    for name, transform in scenarios.items():
        loader = dataloader_factory(transform)
        res = evaluate_accuracy(model, loader, device=device)
        results[name] = res["accuracy"]

    return results


# --------------------------------------------------------------------------- #
# 7. GRAD-CAM
# --------------------------------------------------------------------------- #
def plot_gradcam(
    model: nn.Module,
    target_layer,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    device="cpu",
    save_path=None,
):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    import matplotlib.pyplot as plt

    model = model.to(device).eval()
    image_tensor = image_tensor.to(device)

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor)[0]
    visualization = show_cam_on_image(
        original_image, grayscale_cam, use_rgb=True
    )

    plt.imshow(visualization)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return visualization


# --------------------------------------------------------------------------- #
# 8. BÁO CÁO TỔNG HỢP
# --------------------------------------------------------------------------- #
def full_report(
    model: nn.Module,
    model_name: str = "Model",
    input_size=(1, 3, 32, 32),
    device=None,
    accuracy: float = None,
) -> dict:
    if device is None:
        device = auto_device()

    print(f"\n{'=' * 60}")
    print(f"BÁO CÁO ĐÁNH GIÁ: {model_name}  (device={device})")
    print(f"{'=' * 60}")

    # 1. Params
    params_info = count_parameters(model)
    print(f"\n[1] Tham số:")
    print(
        f"    Tổng           : {params_info['total_params']:,} "
        f"({params_info['size_m']:.3f} M)"
    )
    print(f"    Trainable      : {params_info['trainable_params']:,}")
    print(f"    Kích thước     : {params_info['size_mb']:.3f} MB")

    breakdown = count_by_top_module(model)
    if breakdown:
        print(f"    Phân bố module :")
        for name, n in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            pct = (
                100 * n / params_info["total_params"]
                if params_info["total_params"]
                else 0
            )
            print(f"      - {name:<20s}: {n:>10,} ({pct:5.1f}%)")

    # 2. FLOPs
    try:
        flops_info = count_flops(model, input_size=input_size, device=device)
        print(f"\n[2] FLOPs:")
        print(
            f"    {flops_info['flops_m']:.3f} M  (phương pháp: {flops_info['method']})"
        )
    except Exception as e:
        flops_info = None
        print(
            f"\n[2] FLOPs: Không tính được\n    {_friendly_cuda_error(e, device)}"
        )

    # 3. Inference time
    try:
        time_info = measure_inference_time(
            model, input_size=input_size, device=device
        )
        print(f"\n[3] Thời gian suy luận ({device}):")
        print(
            f"    Trung bình     : {time_info['mean_ms']:.3f} ms "
            f"(± {time_info['std_ms']:.3f}, {time_info['n_runs']} lần chạy)"
        )
    except RuntimeError as e:
        time_info = None
        print(f"\n[3] Thời gian suy luận: LỖI\n    {e}")

    # 4. IDS (nếu có accuracy)
    ids_value = None
    if accuracy is not None:
        ids_value = compute_ids(accuracy, params_info["size_m"])
        print(f"\n[4] Information Density (IDS):")
        print(f"    ACC = {accuracy:.2f}%  ->  IDS = {ids_value:.2f}")

    return {
        "model_name": model_name,
        "params": params_info,
        "flops": flops_info,
        "inference_time": time_info,
        "accuracy": accuracy,
        "ids": ids_value,
    }


# --------------------------------------------------------------------------- #
# 9. KHỞI TẠO MÔ HÌNH (BUILD MODEL)
# --------------------------------------------------------------------------- #
def build_Model(name, num_classes=43, pretrained=False):
    """Hàm tạo mô hình theo tên tiêu chuẩn."""
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

    # 10 Mô hình Benchmark chuẩn
    elif name in ["VGG16", "VGG-16"]:
        return timm.create_model(
            "vgg16", pretrained=pretrained, num_classes=num_classes
        )
    elif name in ["RESNET18", "ResNet18"]:
        return timm.create_model(
            "resnet18", pretrained=pretrained, num_classes=num_classes
        )
    elif name in ["VIT_B", "ViT-B"]:
        return timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=32,
        )
    elif name in ["VIT_S", "ViT-S"]:
        return timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=32,
        )
    elif name in ["EFFICIENTNET_B0", "EfficientNet-B0"]:
        return timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=num_classes
        )
    elif name in ["MOBILENETV3_SMALL", "MobileNetV3-Small"]:
        return timm.create_model(
            "mobilenetv3_small_100",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif name in ["GHOSTNET", "GhostNet"]:
        return timm.create_model(
            "ghostnet_100", pretrained=pretrained, num_classes=num_classes
        )
    else:
        raise ValueError(f"Tên mô hình '{name}' không tồn tại.")


# --------------------------------------------------------------------------- #
# MAIN BENCHMARK LOOP
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    device = auto_device()
    num_classes = 43
    input_size = (1, 3, 32, 32)

    print("=" * 80)
    print(f"BẮT ĐẦU BENCHMARK TOÀN BỘ MÔ HÌNH TRÊN DEVICE: {device}")
    print("=" * 80)

    # Danh sách tất cả các mô hình trong build_Model và Accuracy tương ứng (nếu có)
    # Các mô hình benchmark nếu chưa có độ chính xác cụ thể có thể để None
    target_models = {
        "LIGHT_HYBRIC_MAMBA": 97.46,
        "MEDIUM_HYBRIC_MAMBA": 97.86,
        "HEAVY_HYBRIC_MAMBA": 98.55,
        "SUPER_MAMBA_DEPT_3": 98.06,
        "SUPER_MAMBA_DEPT_4": 98.43,
        "VGG16": None,
        "RESNET18": None,
        "VIT_B": None,
        "VIT_S": None,
        "EFFICIENTNET_B0": None,
        "MOBILENETV3_SMALL": None,
        "GHOSTNET": None,
    }

    final_results = {}
    reports = {}

    for model_name, acc in target_models.items():
        print(f"\n>>> Đang thực thi benchmark cho: {model_name} ...")

        try:
            # 1. Khởi tạo mô hình
            model = build_Model(
                model_name, num_classes=num_classes, pretrained=False
            )

            # 2. Chạy báo cáo tổng quan (Params, FLOPs, Single Inference, IDS)
            rep = full_report(
                model,
                model_name=model_name,
                input_size=input_size,
                device=device,
                accuracy=acc,
            )
            reports[model_name] = rep

            # 3. Đo độ trễ suy luận nhiều lần (10 runs x 400 iterations)
            latencies = []
            for _ in range(10):
                t_info = measure_inference_time(
                    model,
                    input_size=input_size,
                    device=device,
                    n_warmup=30,
                    n_runs=200,
                )
                latencies.append(t_info["median_ms"])
                time.sleep(0.2)

            final_results[model_name] = latencies

        except Exception as e:
            print(f"❌ XẢY RA LỖI KHI ĐO MÔ HÌNH {model_name}: {e}")
            final_results[model_name] = None

        finally:
            # Dọn dẹp GPU memory tránh OOM giữa các mô hình
            if "model" in locals():
                del model
            if device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        time.sleep(1)

    # ----------------------------------------------------------------------- #
    # TỔNG HỢP KẾT QUẢ VÀ IN BẢNG BÁO CÁO
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 90)
    print("TỔNG HỢP KẾT QUẢ BENCHMARK TOÀN BỘ MÔ HÌNH (LATENCY OVER 10 RUNS)")
    print("=" * 90)
    print(
        f"{'Model Name':<22s} | {'Params (M)':<10s} | {'FLOPs (M)':<10s} | {'Median Latency (ms)':<20s} | {'IDS':<8s}"
    )
    print("-" * 90)

    for name in target_models.keys():
        times = final_results.get(name)
        rep = reports.get(name, {})

        params_m = (
            f"{rep['params']['size_m']:.3f}M" if rep.get("params") else "N/A"
        )
        flops_m = (
            f"{rep['flops']['flops_m']:.2f}M" if rep.get("flops") else "N/A"
        )
        ids_str = f"{rep['ids']:.2f}" if rep.get("ids") is not None else "N/A"

        if times:
            med = np.median(times)
            min_t = min(times)
            max_t = max(times)
            latency_str = f"{med:.3f} ms ({min_t:.3f} - {max_t:.3f})"
        else:
            latency_str = "FAILED"

        print(
            f"{name:<22s} | {params_m:<10s} | {flops_m:<10s} | {latency_str:<20s} | {ids_str:<8s}"
        )

    print("=" * 90)