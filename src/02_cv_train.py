"""
02_cv_train.py — 交叉验证多模型训练
======================================
用不同 random_state 切分训练/验证集，训练 3 个独立模型。
每个模型看不同 10% 验证数据 → 错误不重叠 → Ensemble 互补。

当前 3 个种子: 2, 42, 100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import importlib
import os

# 导入原始数据（未切分）
dp = importlib.import_module('01_data_prep_cv')
X_full = dp.X_full
y_full = dp.y_full

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ============================================================
# 通用配置
# ============================================================
BATCH_SIZE = 86
EPOCHS = 30
PATIENCE = 5

RANDOM_STATES = [2, 42, 100]  # 3 个不同切分

train_transform = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0),
])

# ============================================================
# 模型定义
# ============================================================
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(7 * 7 * 64, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, 10),
        )
    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.classifier(x)
        return x

# ============================================================
# 为每个种子训练一个模型
# ============================================================
for rs in RANDOM_STATES:
    print(f"\n{'='*60}")
    print(f"训练 random_state={rs}")
    print(f"{'='*60}")

    # 切分
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_full, y_full, test_size=0.1, random_state=rs
    )

    # Dataset + DataLoader (train 带增强，val 不带)
    class MNISTDataset(TensorDataset):
        def __init__(self, X, y, transform=None):
            super().__init__(torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2),
                             torch.tensor(y, dtype=torch.long))
            self.transform = transform
        def __getitem__(self, idx):
            img, label = super().__getitem__(idx)
            if self.transform: img = self.transform(img)
            return img, label

    train_ds = MNISTDataset(X_tr, y_tr, transform=train_transform)
    val_ds   = MNISTDataset(X_va, y_va, transform=None)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 模型 + 优化器
    model = DigitCNN().to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("../outputs", exist_ok=True)
            torch.save(model.state_dict(), f"../outputs/model_rs{rs}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE: break

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:2d}/{EPOCHS} | train_loss: {train_loss:.4f} | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | lr: {lr:.6f}")

    print(f"✅ random_state={rs} 完成，最优 val_acc={best_val_acc:.4f}")

print(f"\n{'='*60}")
print("全部训练完成！")
for rs in RANDOM_STATES:
    print(f"  outputs/model_rs{rs}.pth")
