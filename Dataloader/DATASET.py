import argparse
import csv
import logging
import os
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm




def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]


IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.ppm')


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
        elif dn in ["belgium", "belgium_split", "belgium_1t1t", "belgium_1train_1test"]:
            self.samples, self.class_to_idx = self._scan_belgium(root)
        else:  # "german", "folders", "default"
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
            raise FileNotFoundError(
                f"Không tìm thấy thư mục Train/Test trong: {root}\n"
                f"Cần có '{root}/Train' và '{root}/Test'."
            )

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

    # ---------- Dataset interface ----------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_name = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[class_name]
        return img, label