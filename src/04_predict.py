"""
04_predict.py — 阶段四：预测测试集 + 生成提交文件
===================================================
参考：Yassine Ghouzam "Introduction to CNN Keras - 0.997 (top 8%)"
对照教程 Section 5.1 (Predict and Submit results)

功能：
  4.1 加载最优模型 + 测试集
  4.2 推理测试集 28,000 张图
  4.3 生成 Kaggle 提交文件 submission.csv
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import importlib
import os
import sys

# ============================================================
# 项目根目录（所有产出文件路径基于此）
# ============================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

# ============================================================
# 导入阶段一的测试集数据
# ============================================================
dp = importlib.import_module('01_data_prep')
test_loader = dp.test_loader      # DataLoader，batch_size=86，无增强

# ============================================================
# 4.0 重新定义模型（必须与 02_model.py 一致）
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
# 4.1 加载训练好的模型
# ============================================================
model_path = os.path.join(OUTPUTS_DIR, "model_best.pth")
if not os.path.exists(model_path):
    print(f"错误：未找到模型文件 {model_path}，请先运行 02_model.py 训练")
    sys.exit(1)

model = DigitCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()
print(f"✅ 已加载模型: {model_path}")
print(f"   设备: {device}")

# ============================================================
# 4.2 推理测试集
# ============================================================
# 教程写法: results = model.predict(test)
#           results = np.argmax(results, axis=1)
# PyTorch 写法: 逐 batch 推理 → argmax（无需 softmax，argmax 结果等价）
all_preds = []

with torch.no_grad():
    for images in test_loader:
        # test_loader 返回的是单个 tensor（无标签），也可能返回 (images,)
        if isinstance(images, (list, tuple)):
            images = images[0]
        images = images.to(device)
        outputs = model(images)
        # 教程: np.argmax(Y_pred, axis=1) —— 取概率最大的类别
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu().numpy())

results = np.concatenate(all_preds)
print(f"✅ 推理完成: {len(results)} 张测试图片")
print(f"   预测类别分布: {np.bincount(results)}")

# ============================================================
# 4.3 生成提交文件
# ============================================================
# 教程格式:
#   submission = pd.concat([
#       pd.Series(range(1, 28001), name="ImageId"),
#       results
#   ], axis=1)
#   submission.to_csv("cnn_mnist_datagen.csv", index=False)

submission = pd.DataFrame({
    "ImageId": range(1, len(results) + 1),
    "Label": results
})

submission_path = os.path.join(OUTPUTS_DIR, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"✅ 提交文件已保存: {submission_path}")
print(f"   行数: {len(submission)}, 列: ImageId, Label")
print(f"\n   文件预览:")
print(submission.head(10).to_string(index=False))
