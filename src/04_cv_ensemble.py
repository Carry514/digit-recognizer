"""
04_cv_ensemble.py — 交叉验证集成预测
======================================
加载 3 个不同 random_state 训练的模型，平均概率取 argmax。
预期比 2 模型集成有更大提升（错误互补性更强）。
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

dp = importlib.import_module('01_data_prep_cv')
test_loader = dp.test_loader

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

# 3 个不同 random_state 的模型
MODEL_NAMES = [
    ("model_rs2.pth",   "random_state=2"),
    ("model_rs42.pth",  "random_state=42"),
    ("model_rs100.pth", "random_state=100"),
]

models = []
for path, label in MODEL_NAMES:
    full = os.path.join(OUTPUTS_DIR, path)
    if not os.path.exists(full):
        print(f"⚠️  跳过 {label}（文件不存在）")
        continue
    m = DigitCNN().to(device)
    m.load_state_dict(torch.load(full, map_location=device, weights_only=True))
    m.eval()
    models.append((m, label))
    print(f"✅ 加载 {path} ({label})")

print(f"   共 {len(models)} 个模型参与集成\n")

# Ensemble 推理
all_preds = []
with torch.no_grad():
    for images in test_loader:
        if isinstance(images, (list, tuple)): images = images[0]
        images = images.to(device)
        bs = images.size(0)

        all_probs = torch.zeros(bs, 10, device=device)
        for model, _ in models:
            all_probs += F.softmax(model(images), dim=1)

        preds = torch.argmax(all_probs / len(models), dim=1)
        all_preds.append(preds.cpu().numpy())

results = np.concatenate(all_preds)
print(f"✅ Ensemble 推理完成: {len(results)} 张")
print(f"   预测类别分布: {np.bincount(results)}")

# 提交
submission = pd.DataFrame({"ImageId": range(1, len(results) + 1), "Label": results})
submission_path = os.path.join(OUTPUTS_DIR, "submission_cv.csv")
submission.to_csv(submission_path, index=False)
print(f"\n✅ 提交文件已保存: {submission_path}")
print(submission.head(10).to_string(index=False))
