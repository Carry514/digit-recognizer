"""
03_evaluate.py — 阶段三：模型评估
===================================
参考：Yassine Ghouzam "Introduction to CNN Keras - 0.997 (top 8%)"
对照教程 Section 4.2 (混淆矩阵) + 4.3 (错误样本展示)

功能：
  3.1 加载最优模型 + 验证集
  3.2 混淆矩阵（sklearn + seaborn）
  3.3 预测错误最严重的 6 张图片展示
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import importlib
import os
import sys

# 项目根目录（所有产出文件路径基于此）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

# ============================================================
# 导入阶段一的验证集数据（01_data_prep.py 已自动算路径，无需 chdir）
# ============================================================
dp = importlib.import_module('01_data_prep')
X_val_raw = dp.X_val   # 原始 numpy 数组 (shape: N, 28, 28, 1)，用于绘图
y_val_raw = dp.y_val   # 整数标签 0-9
val_loader = dp.val_loader

# ============================================================
# 3.0 重新定义模型（必须与 02_model.py 一致）
# ============================================================
class DigitCNN(nn.Module):
    """与 02_model.py 完全相同的架构"""
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 3.1 加载训练好的模型
# ============================================================
model_path = os.path.join(OUTPUTS_DIR, "model_acc_best.pth")
if not os.path.exists(model_path):
    print(f"错误：未找到模型文件 {model_path}，请先运行 02_model.py 训练")
    sys.exit(1)

model = DigitCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()
print(f"✅ 已加载模型: {model_path}")
print(f"   设备: {device}")

# ============================================================
# 3.2 在验证集上推理
# ============================================================
# 教程写法: Y_pred = model.predict(X_val)
# PyTorch 写法: 逐 batch 推理，拼接结果
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)        # 教程: np.argmax(Y_pred, axis=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())

y_pred_classes = np.concatenate(all_preds)
y_true = np.concatenate(all_labels)

# 计算准确率
accuracy = (y_pred_classes == y_true).mean()
print(f"   验证集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ============================================================
# 3.3 混淆矩阵（教程 Section 4.2）
# ============================================================
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix — Validation Set", fontsize=14)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "confusion_matrix_acc.png"), dpi=150)
plt.show()
print("✅ 混淆矩阵已保存到 outputs/confusion_matrix.png")

# ============================================================
# 3.4 错误分析（教程 Section 4.3）
# ============================================================
# 找预测错误的样本
errors_idx = (y_pred_classes != y_true)
y_pred_errors = y_pred_classes[errors_idx]
y_true_errors = y_true[errors_idx]

num_errors = len(y_pred_errors)
print(f"\n   验证集错误数: {num_errors} / {len(y_true)} ({num_errors/len(y_true)*100:.2f}%)")

if num_errors == 0:
    print("🎉 完美！验证集上无任何错误。")
    sys.exit(0)

# 获取错误样本对应的概率
# 教程中 model.predict 输出概率，我们用 softmax 得到
all_probs = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())

Y_pred_prob = np.concatenate(all_probs)           # (N, 10) 概率矩阵
Y_pred_prob_errors = Y_pred_prob[errors_idx]      # 错误样本的概率

# 教程: 预测标签的最高概率
Y_pred_errors_prob = np.max(Y_pred_prob_errors, axis=1)

# 教程: 真实标签的概率
true_prob_errors = np.array([
    Y_pred_prob_errors[i, y_true_errors[i]] for i in range(num_errors)
])

# 教程: 差值（越大说明模型对错误答案越"自信"）
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# 找出差值最大的 6 个（教程: sorted_dela_errors[-6:]）
sorted_delta_errors = np.argsort(delta_pred_true_errors)
most_important_errors = sorted_delta_errors[-6:]

# ============================================================
# 3.5 绘制错误样本（教程 display_errors 函数）
# ============================================================
# 教程中 X_val 是 (N, 28, 28, 1)，绘图前 reshape 为 (28, 28)
X_val_errors = X_val_raw[errors_idx]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle("Top 6 Most Confident Mistakes", fontsize=16, fontweight='bold')
axes = axes.flatten()

for i, err_idx in enumerate(most_important_errors):
    ax = axes[i]
    # 教程: img_errors[error].reshape((28,28))
    ax.imshow(X_val_errors[err_idx].reshape(28, 28), cmap="gray")
    ax.set_title(f"Pred: {y_pred_errors[err_idx]} | True: {y_true_errors[err_idx]}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, "error_samples_acc.png"), dpi=150)
plt.show()
print("✅ 错误样本已保存到 outputs/error_samples.png")

# 汇总
print("\n" + "=" * 60)
print("📊 评估完成")
print(f"   验证准确率: {accuracy*100:.2f}%")
print(f"   错误样本数: {num_errors} / {len(y_true)}")
print(f"   产出文件: outputs/confusion_matrix.png, outputs/error_samples.png")
print("=" * 60)
