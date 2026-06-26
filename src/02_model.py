"""
02_model.py — 阶段二：CNN 建模 + 训练
========================================
参考：Yassine Ghouzam "Introduction to CNN Keras - 0.997 (top 8%)"
架构完全对应教程，仅框架翻译为 PyTorch。

功能：
  2.1 定义 CNN 模型（与教程架构一致）
  2.2 配置优化器 (RMSprop) + 学习率调度 (ReduceLROnPlateau)
  2.3 训练循环（30 epochs, EarlyStopping）
  2.4 保存最优模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import importlib
import os

# 导入阶段一的数据（文件名以数字开头，需用 importlib）
dp = importlib.import_module('01_data_prep')
train_loader = dp.train_loader
val_loader = dp.val_loader

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ============================================================
# 2.1 定义 CNN 模型
# ============================================================
# 架构对应教程：
#   In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2
#      -> Flatten -> Dense -> Dropout -> Out

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # 第一个卷积块：32 通道，kernel=5，输出 28→14
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),   # 28x28x1  → 28x28x32
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),  # 28x28x32 → 28x28x32
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 28x28x32 → 14x14x32
            nn.Dropout(0.25),
        )

        # 第二个卷积块：64 通道，kernel=3，输出 14→7
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 14x14x32 → 14x14x64
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   # 14x14x64 → 14x14x64
            nn.ReLU(),
            nn.MaxPool2d(2),                                # 14x14x64 → 7x7x64
            nn.Dropout(0.25),
        )

        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 256),   # 3136 → 256
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),            # 256 → 10
            # 无需 Softmax，CrossEntropyLoss 内置
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x


model = DigitCNN().to(device)
print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# 2.2 配置优化器和学习率调度
# ============================================================
# 优化器：与教程一致，RMSprop (lr=0.001)
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# ReduceLROnPlateau：与教程一致，监控 val_acc（准确率不升 → lr 减半）
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='max',            # 监控指标越大越好（准确率）
    factor=0.5,            # lr *= 0.5
    patience=3,            # 连续 3 epoch 不升
    min_lr=1e-5,           # 最低学习率
)

# 损失函数（内置 Softmax，无需 one-hot）
criterion = nn.CrossEntropyLoss()


# ============================================================
# 2.3 训练循环
# ============================================================
EPOCHS = 30
PATIENCE = 5            # EarlyStopping 耐心值

best_val_loss = float('inf')
patience_counter = 0

# 记录历史，用于画曲线
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

print(f"\n开始训练 ({EPOCHS} epochs, EarlyStopping patience={PATIENCE})")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):
    # ---- 训练阶段 ----
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)

    # ---- 验证阶段 ----
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = correct / total

    # 记录历史
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # 学习率调度（监控验证准确率）
    scheduler.step(val_acc)

    # 打印进度
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:2d}/{EPOCHS} | "
          f"train_loss: {train_loss:.4f} | "
          f"val_loss: {val_loss:.4f} | "
          f"val_acc: {val_acc:.4f} | "
          f"lr: {current_lr:.6f}")

    # ---- EarlyStopping + 保存最优模型 ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        os.makedirs("../outputs", exist_ok=True)
        torch.save(model.state_dict(), "../outputs/model_best.pth")
        print(f"  ✅ 保存最优模型 (val_loss={best_val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹ EarlyStopping: val_loss 连续 {PATIENCE} epoch 未改善")
            break

print("=" * 60)
print(f"训练结束。最优 val_loss = {best_val_loss:.4f}")


# ============================================================
# 2.4 绘制训练曲线
# ============================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

ax1.plot(history['train_loss'], color='b', label='Training loss')
ax1.plot(history['val_loss'], color='r', label='Validation loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend(loc='best')
ax1.set_title('Training and Validation Loss')

ax2.plot(history['val_acc'], color='g', label='Validation accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend(loc='best')
ax2.set_title('Validation Accuracy')

plt.tight_layout()
plt.savefig("../outputs/training_curves.png", dpi=150)
plt.show()
print("训练曲线已保存到 outputs/training_curves.png")
