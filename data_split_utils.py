

import json
import os

from sklearn.model_selection import train_test_split


def _split_file_path(save_path: str, dataset_name: str, seed: int) -> str:
    split_dir = os.path.join(save_path, "_dataset_splits")
    os.makedirs(split_dir, exist_ok=True)
    return os.path.join(split_dir, f"{dataset_name}_split_seed{seed}.json")


def get_or_create_split(full_dataset, save_path: str, dataset_name: str,
                         seed: int = 2223, force_recreate: bool = False):

    split_path = _split_file_path(save_path, dataset_name, seed)

    if os.path.exists(split_path) and not force_recreate:
        print(f"[SPLIT] ✅ Dùng lại split cố định đã lưu: {split_path}")
        with open(split_path, "r", encoding="utf-8") as f:
            split_info = json.load(f)

        path_to_sample = {p: (p, c) for p, c in full_dataset.samples}

        def _resolve(path_list, part_name):
            resolved, missing = [], 0
            for p in path_list:
                if p in path_to_sample:
                    resolved.append(path_to_sample[p])
                else:
                    missing += 1
            if missing > 0:
                print(f"  ⚠️ CẢNH BÁO: {missing} ảnh trong '{part_name}' đã lưu "
                      f"không còn tồn tại trong dataset hiện tại (dữ liệu gốc có "
                      f"thể đã thay đổi/xoá bớt).")
            return resolved

        train_samples = _resolve(split_info["train_paths"], "train")
        val_samples = _resolve(split_info["val_paths"], "val")
        test_samples = _resolve(split_info["test_paths"], "test")

        print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
        return train_samples, val_samples, test_samples

    # ---------- Chưa có -> tạo mới 1 LẦN DUY NHẤT rồi lưu lại ----------
    print(f"[SPLIT] ⚙️ Chưa có file split cố định, đang tạo mới tại: {split_path}")

    indices = list(range(len(full_dataset)))
    labels = [full_dataset.class_to_idx[full_dataset.samples[i][1]] for i in indices]

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, shuffle=True, stratify=labels
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, shuffle=True, stratify=temp_labels
    )

    train_samples = [full_dataset.samples[i] for i in train_idx]
    val_samples = [full_dataset.samples[i] for i in val_idx]
    test_samples = [full_dataset.samples[i] for i in test_idx]

    # ---------- Sanity check: đảm bảo tuyệt đối không trùng lặp ----------
    train_paths = set(s[0] for s in train_samples)
    val_paths = set(s[0] for s in val_samples)
    test_paths = set(s[0] for s in test_samples)

    overlap_train_val = train_paths & val_paths
    overlap_train_test = train_paths & test_paths
    overlap_val_test = val_paths & test_paths

    assert len(overlap_train_val) == 0, f"Lỗi nghiêm trọng: {len(overlap_train_val)} ảnh trùng giữa train/val!"
    assert len(overlap_train_test) == 0, f"Lỗi nghiêm trọng: {len(overlap_train_test)} ảnh trùng giữa train/test!"
    assert len(overlap_val_test) == 0, f"Lỗi nghiêm trọng: {len(overlap_val_test)} ảnh trùng giữa val/test!"

    print(f"  ✅ Sanity check: 0 trùng lặp giữa train/val/test.")
    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    split_info = {
        "seed": seed,
        "dataset_name": dataset_name,
        "train_paths": [s[0] for s in train_samples],
        "val_paths": [s[0] for s in val_samples],
        "test_paths": [s[0] for s in test_samples],
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f)
    print(f"  📄 Đã lưu split cố định, các lần chạy sau (train lại, benchmark...) "
          f"sẽ tự động dùng lại đúng bộ này.")

    return train_samples, val_samples, test_samples