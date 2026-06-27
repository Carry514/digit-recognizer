"""
04_ensemble.py — 多模型集成预测
==================================
加载多个训练好的模型，平均预测概率取 argmax。
模型间的错误通常不相关 → 平均后互相修正，提升 0.1-0.3%

当前集成:
  - model_best.pth     (val_loss 最优, Kaggle 0.99482)
  - model_acc_best.pth (val_acc 最优, Kaggle 0.99450)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import importlib
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

# 导入测试集
dp = importlib.import_module('01_data_prep')
test_loader = dp.test_loader

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 加载多个模型
# ============================================================
MODEL_PATHS = [
    "model_best.pth",     # 第一次训练，Kaggle 0.99482
    "model_acc_best.pth", # 第三次训练，val_acc 最优保存
]

models = []
for name in MODEL_PATHS:
    path = os.path.join(OUTPUTS_DIR, name)
    if not os.path.exists(path):
        print(f"⚠️  跳过 {name}（文件不存在）")
        continue
    m = DigitCNN().to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.eval()
    models.append(m)
    print(f"✅ 加载 {name}")

if len(models) < 2:
    print("错误：至少需要 2 个模型做集成")
    sys.exit(1)

print(f"   共 {len(models)} 个模型参与集成\n")

# ============================================================
# Ensemble 推理
# ============================================================
all_preds = []

with torch.no_grad():
    for images in test_loader:
        if isinstance(images, (list, tuple)):
            images = images[0]
        images = images.to(device)
        batch_size = images.size(0)

        # 每个模型做预测，取概率
        all_probs = torch.zeros(batch_size, 10, device=device)
        for model in models:
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            all_probs += probs

        # 平均概率 → argmax
        avg_probs = all_probs / len(models)
        preds = torch.argmax(avg_probs, dim=1)
        all_preds.append(preds.cpu().numpy())

results = np.concatenate(all_preds)
print(f"✅ Ensemble 推理完成: {len(results)} 张")
print(f"   预测类别分布: {np.bincount(results)}")

# 对比单个模型
print(f"\n   模型数: {len(models)}")
for i, name in enumerate(MODEL_PATHS[:len(models)]):
    single_preds = []
    with torch.no_grad():
        for images in test_loader:
            if isinstance(images, (list, tuple)): images = images[0]
            images = images.to(device)
            preds = torch.argmax(models[i](images), dim=1)
            single_preds.append(preds.cpu().numpy())
    single = np.concatenate(single_preds)
    agree = (single == results).mean()
    print(f"   {name}: 与 Ensemble 一致率 = {agree:.4f}")

# ============================================================
# 生成提交文件
# ============================================================
submission = pd.DataFrame({"ImageId": range(1, len(results) + 1), "Label": results})
submission_path = os.path.join(OUTPUTS_DIR, "submission_ensemble.csv")
submission.to_csv(submission_path, index=False)
print(f"\n✅ 提交文件已保存: {submission_path}")
print(f"   文件预览:")
print(submission.head(10).to_string(index=False))
