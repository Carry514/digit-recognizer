"""
01_data_prep_cv.py — 数据准备（CV 交叉验证版）
=================================================
与 01_data_prep.py 相同，额外导出 X_full / y_full（未切分的原始数据），
供 02_cv_train.py 用自己的 random_state 切分。
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
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_SCRIPT_DIR), "data") + os.sep
train_df = pd.read_csv(DATA_DIR + "train.csv")
test_df  = pd.read_csv(DATA_DIR + "test.csv")


# ============================================================
# 1.2 归一化 + 重塑
# ============================================================
y = train_df["label"].values
X = train_df.drop("label", axis=1).values
X_test = test_df.values

X = X / 255.0
X_test = X_test / 255.0

X = X.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# ⭐ 导出未切分的全量数据
X_full = X
y_full = y

# ============================================================
# 1.3 拆分训练/验证集（默认 random_state=2，与教程一致）
# ============================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=2
)

# ============================================================
# 1.4 构建 DataLoader（原版脚本不用这些，但保持接口兼容）
# ============================================================
class MNISTDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1)
        if self.transform: img = self.transform(img)
        if self.labels is not None:
            return img, torch.tensor(self.labels[idx], dtype=torch.long)
        return img

train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0),
])
val_test_transform = None
BATCH_SIZE = 86

train_dataset = MNISTDataset(X_train, y_train, transform=train_transform)
val_dataset   = MNISTDataset(X_val,   y_val,   transform=val_test_transform)
test_dataset  = MNISTDataset(X_test,  labels=None, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

if __name__ == "__main__":
    print(f"全量数据: X_full={X_full.shape}, y_full={y_full.shape}")
    print(f"训练集:   X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"验证集:   X_val={X_val.shape},   y_val={y_val.shape}")
    print(f"测试集:   X_test={X_test.shape}")
    print("✅ 01_data_prep_cv 加载完成")
