"""
01_data_prep.py — 阶段一：数据准备
=====================================
参考：Yassine Ghouzam "Introduction to CNN Keras - 0.997 (top 8%)"
功能：
  1.1 加载 train.csv
  1.2 检查缺失值
  1.3 归一化 (0-255 → 0-1)
  1.4 重塑为 (28, 28, 1)
  1.5 拆分训练集 / 验证集 (90% / 10%)
  1.6 构建 DataLoader（训练集带数据增强）
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# 1.1 加载数据
# ============================================================
# 根据脚本文件位置计算 data/ 目录（不论从哪里 import 都能正确找到）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data") + os.sep
train_df = pd.read_csv(DATA_DIR + "train.csv")
test_df  = pd.read_csv(DATA_DIR + "test.csv")

if __name__ == "__main__":
    print("=" * 50)
    print("1.1 加载数据")
    print(f"  训练集: {train_df.shape}")
    print(f"  测试集: {test_df.shape}")


# ============================================================
# 1.2 检查缺失值
# ============================================================
if __name__ == "__main__":
    print("\n1.2 检查缺失值")
    print(f"  训练集缺失值总数: {train_df.isnull().any().describe()}")
    print(f"  测试集缺失值总数: {test_df.isnull().any().describe()}")


# ============================================================
# 1.3 归一化
# ============================================================
# 分离特征和标签
y = train_df["label"].values                      # (42000,) 整数 0-9
X = train_df.drop("label", axis=1).values         # (42000, 784)
X_test = test_df.values                           # (28000, 784)

# 归一化到 [0, 1]
X = X / 255.0
X_test = X_test / 255.0

if __name__ == "__main__":
    print("\n1.3 归一化")
    print(f"  X 范围: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  X_test 范围: [{X_test.min():.2f}, {X_test.max():.2f}]")


# ============================================================
# 1.4 重塑维度：784 → (28, 28, 1)
# ============================================================
X = X.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

if __name__ == "__main__":
    print("\n1.4 重塑维度")
    print(f"  X 形状: {X.shape}")
    print(f"  X_test 形状: {X_test.shape}")


# ============================================================
# 1.5 拆分训练集 / 验证集 (90/10, random_state=2)
# ============================================================
# 注意：PyTorch 不需要 one-hot 编码，直接用整数标签
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.1,
    random_state=2          # 与 Ghouzam 教程一致
)

if __name__ == "__main__":
    print("\n1.5 拆分训练/验证集")
    print(f"  训练集 X: {X_train.shape}, y: {y_train.shape}")
    print(f"  验证集 X: {X_val.shape},   y: {y_val.shape}")


# ============================================================
# 1.6 构建 DataLoader
# ============================================================

class MNISTDataset(Dataset):
    """将 numpy 数组包装成 PyTorch Dataset"""
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # HWC → CHW（PyTorch 标准格式）
        img = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1)
        if self.transform:
            img = self.transform(img)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label
        else:
            return img


# 训练集增强 —— 严格对应 Ghouzam 教程的 ImageDataGenerator 参数
# rotation_range=10, zoom_range=0.1, width/height_shift_range=0.1
# 教程未使用 shear，未使用额外标准化（仅 /255.0）
train_transform = transforms.Compose([
    transforms.RandomAffine(
        degrees=10,              # rotation_range=10
        translate=(0.1, 0.1),    # width/height_shift_range=0.1
        scale=(0.9, 1.1),        # zoom_range=0.1
        shear=0                  # 教程未使用 shear
    ),
])

# 验证集 / 测试集不做增强（教程中 validation_data 直接使用原始数据）
val_test_transform = None

BATCH_SIZE = 86  # 与教程一致

train_dataset = MNISTDataset(X_train, y_train, transform=train_transform)
val_dataset   = MNISTDataset(X_val,   y_val,   transform=val_test_transform)
test_dataset  = MNISTDataset(X_test,  labels=None, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

if __name__ == "__main__":
    print("\n1.6 构建 DataLoader")
    print(f"  train_loader: {len(train_loader)} batches")
    print(f"  val_loader:   {len(val_loader)} batches")
    print(f"  test_loader:  {len(test_loader)} batches")

    # 验证一个 batch
    sample_img, sample_label = next(iter(train_loader))
    print(f"\n  验证 batch: 图像形状 = {sample_img.shape}, 标签形状 = {sample_label.shape}")
    print(f"  标签范围: [{sample_label.min()}, {sample_label.max()}]")
    print("=" * 50)
