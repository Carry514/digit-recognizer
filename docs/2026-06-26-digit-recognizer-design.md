# Digit Recognizer 课程设计 — 设计方案

## 基本信息

| 项目 | 说明 |
|------|------|
| 题目 | Kaggle Digit Recognizer |
| 任务 | 手写数字 0-9 识别（MNIST） |
| 语言/框架 | Python + PyTorch |
| 完成方式 | 独立完成 |
| 目标成绩 | A（排名前 14%） |
| 目标准确率 | ≥ 99.4% |

## 方案选择：增强 CNN + 数据增强

采用方案 B，在经典 CNN 基础上加入 BatchNorm、Dropout、数据增强，平衡实现复杂度与排名的投入产出比。

## 模型架构

```
Input (28x28x1)
  → Conv2d(1→64, 3x3, padding=1) → BatchNorm2d → ReLU
  → Conv2d(64→64, 3x3, padding=1) → BatchNorm2d → ReLU
  → MaxPool2d(2x2) → Dropout(0.25)

  → Conv2d(64→128, 3x3, padding=1) → BatchNorm2d → ReLU
  → Conv2d(128→128, 3x3, padding=1) → BatchNorm2d → ReLU
  → MaxPool2d(2x2) → Dropout(0.25)

  → Conv2d(128→256, 3x3, padding=1) → BatchNorm2d → ReLU
  → MaxPool2d(2x2) → Dropout(0.25)

  → Flatten → FC(256*3*3→512) → BatchNorm1d → ReLU → Dropout(0.5)
  → FC(512→10) → Softmax
```

- 参数量：约 300 万
- 3 个卷积块 + 2 个全连接层
- 每个卷积块：双层 Conv → BN → ReLU → Pooling → Dropout

## 数据增强（训练时）

| 增强方式 | 参数 | 目的 |
|----------|------|------|
| 随机旋转 | ±10° | 模拟手写角度变化 |
| 随机平移 | ±10% | 模拟书写位置偏移 |
| 随机缩放 | 90%-110% | 模拟书写大小变化 |
| 随机剪切 | ±5° | 模拟倾斜 |
| 归一化 | μ=0.1307, σ=0.3081 | MNIST 标准归一化 |

## 训练策略

| 项目 | 设置 |
|------|------|
| 优化器 | Adam (lr=0.001, 后期 ReduceLROnPlateau) |
| 损失函数 | CrossEntropyLoss |
| Batch Size | 64 |
| Epochs | 30（EarlyStopping patience=5） |
| 训练/验证拆分 | 90% / 10%（从 train.csv 拆分） |
| 学习率衰减 | val_loss 连续 3 epoch 不降则 lr *= 0.5 |

## 项目结构

```
digit-recognizer/
├── data/
│   ├── train.csv          # 训练集（需从 Kaggle 下载）
│   ├── test.csv           # 测试集（需从 Kaggle 下载）
│   └── sample_submission.csv
├── src/
│   ├── dataset.py         # 数据加载 + 增强
│   ├── model.py           # CNN 模型定义
│   ├── train.py           # 训练脚本
│   └── predict.py         # 推理 + 生成提交文件
├── outputs/
│   ├── model_best.pth     # 最优模型权重
│   └── submission.csv     # 最终提交文件
├── docs/
│   └── 2026-06-26-digit-recognizer-design.md  # 本文件
└── README.md
```

## 评分与验收标准

- 提交 Kaggle 获得排名截图
- 独立完成需排名前 14% 可获 A
- 最终答辩需准备：模型原理、训练过程、结果分析

## 依赖

```txt
torch>=2.0
torchvision
pandas
numpy
scikit-learn
matplotlib
tqdm
```
