# Digit Recognizer 课程设计 — 实现方案

> 基于 [Yassine Ghouzam: Introduction to CNN Keras - 0.997 (top 6%)](https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)  
> 原教程为 Keras/TensorFlow，本方案用 PyTorch 翻译实现，核心架构和策略完全一致。

## 基本信息

| 项目 | 说明 |
|------|------|
| 题目 | Kaggle Digit Recognizer |
| 框架 | Python + PyTorch |
| 完成方式 | 独立完成 |
| 目标成绩 | A（排名前 14%） |
| 参考准确率 | 99.7%（top 6%，教程原版） |

---

## 阶段一：数据准备

### 1.1 加载数据
- `pandas.read_csv('train.csv')` 加载 42,000 条数据
- 分离特征 X（784 像素列）和标签 y（label 列）

### 1.2 检查缺失值

- `isnull()` 检查，确认数据完整无缺失

### 1.3 归一化
- 像素值 0-255 → 除以 255.0 → 归一到 [0, 1]

### 1.4 重塑维度
- 784 一维向量 → reshape 为 (28, 28, 1) 图像格式
- 原因：CNN 卷积层需要 (H, W, C) 格式输入

### 1.5 标签格式
- ⚠️ 与教程的差异：PyTorch 的 `CrossEntropyLoss` 内置 softmax，直接使用整数标签 0-9，**不需要 one-hot 编码**

### 1.6 拆分训练/验证集
- `train_test_split`（90% 训练 / 10% 验证）
- `random_state=2`，与教程保持一致
- 验证集用于监控训练过程、EarlyStopping、选择最优模型

---

## 阶段二：CNN 建模

### 2.1 模型架构

| 层 | 类型 | 参数 | 输出尺寸 |
|----|------|------|----------|
| 1 | Conv2d | 1→32, kernel=5×5, padding=2 | 28×28×32 |
|   | ReLU | - | 28×28×32 |
| 2 | Conv2d | 32→32, kernel=5×5, padding=2 | 28×28×32 |
|   | ReLU | - | 28×28×32 |
|   | MaxPool2d | 2×2 | 14×14×32 |
|   | Dropout | p=0.25 | 14×14×32 |
| 3 | Conv2d | 32→64, kernel=3×3, padding=1 | 14×14×64 |
|   | ReLU | - | 14×14×64 |
| 4 | Conv2d | 64→64, kernel=3×3, padding=1 | 14×14×64 |
|   | ReLU | - | 14×14×64 |
|   | MaxPool2d | 2×2 | 7×7×64 |
|   | Dropout | p=0.25 | 7×7×64 |
| 5 | Flatten | - | 3136 |
| 6 | Linear | 3136→256 | 256 |
|   | ReLU | - | 256 |
|   | Dropout | p=0.5 | 256 |
| 7 | Linear | 256→10 | 10 |

> 与教程**完全一致**，仅将 Keras Sequential API 翻译为 PyTorch `nn.Module`。

### 2.2 优化器
- **Adam** (`lr=0.001`, `betas=(0.9, 0.999)`)
- 与教程一致

### 2.3 学习率调度
- **ReduceLROnPlateau**：监控 `val_loss`，连续 3 个 epoch 不降 → lr 减半（factor=0.5）
- 最低 lr = 1e-6
- 与教程一致

### 2.4 数据增强（仅训练集）

| 增强方式 | 参数 | 对应 Keras API |
|----------|------|---------------|
| 随机旋转 | ±10° | `rotation_range=10` |
| 随机平移 | ±10%（相对图像大小） | `width_shift_range=0.1`, `height_shift_range=0.1` |
| 随机缩放 | 90%~110% | `zoom_range=0.1` |
| 随机剪切 | ±5° | `shear_range=5`（近似） |

- 验证集不做增强，仅归一化
- PyTorch 用 `torchvision.transforms` 组合实现
- DataLoader：`batch_size=64`，训练集 shuffle，验证集不 shuffle

---

## 阶段三：模型评估

### 3.1 训练配置
- 损失函数：`CrossEntropyLoss`
- 总 Epochs：30
- EarlyStopping：val_loss 连续 5 轮不降 → 提前停止
- 每轮记录：train_loss, val_loss, val_accuracy
- 自动保存 val_accuracy 最高的模型权重

### 3.2 训练曲线
- 绘制 loss 曲线（train_loss + val_loss 同一张图）
- 绘制 accuracy 曲线
- `matplotlib` 实现，与教程做法一致

### 3.3 混淆矩阵
- 用最优模型预测验证集
- `sklearn.metrics.confusion_matrix`
- `seaborn` 热力图可视化
- 分析哪些数字容易混淆（如 4↔9，3↔5）

### 3.4 错误分析
- 展示预测错误的样本图片
- 标注：真实标签 vs 预测标签
- 分析典型错误模式（答辩加分项）

---

## 阶段四：预测与提交

### 4.1 加载测试集
- `pd.read_csv('test.csv')` 加载 28,000 张测试图
- 归一化 + reshape，与训练集处理流程一致

### 4.2 推理预测
- 切换到 `model.eval()` 模式
- 逐 batch 推理，`torch.no_grad()` 禁用梯度
- 输出每张图的预测类别（0-9）

### 4.3 生成提交文件
- 格式：`ImageId, Label`（与 `sample_submission.csv` 一致）
- 用 pandas 写出 `submission.csv`

---

## 代码文件结构

```
digit-recognizer/
├── data/
│   ├── train.csv                 # 需从 Kaggle 下载
│   ├── test.csv                  # 需从 Kaggle 下载
│   └── sample_submission.csv     # 已下载
├── src/
│   ├── 01_data_prep.py           # 阶段一：数据加载、预处理、拆分
│   ├── 02_model.py               # 阶段二：CNN 模型 + 增强配置 + 训练循环
│   ├── 03_evaluate.py            # 阶段三：训练曲线 + 混淆矩阵 + 错误分析
│   └── 04_predict.py             # 阶段四：推理 + 生成 submission.csv
├── outputs/
│   ├── model_best.pth            # 最优模型权重
│   └── submission.csv            # Kaggle 提交文件
└── docs/
    └── plan.md                   # 本文件
```

---

## 与原教程的差异说明

| 差异点 | 教程（Keras） | 本方案（PyTorch） | 影响 |
|--------|-------------|-----------------|------|
| 标签编码 | `to_categorical` one-hot | 整数标签 0-9 | CrossEntropyLoss 内置 softmax |
| 训练循环 | `model.fit_generator()` 自动 | 手写 for epoch 循环 | 答辩可详细解释训练过程 |
| 数据增强 | `ImageDataGenerator` | `torchvision.transforms` | 功能等价 |
| 模型保存 | `model.save()` | `torch.save()` | 功能等价 |
| 优化器 | RMSprop (lr=0.001) | Adam (lr=0.001) | 性能相近，Adam 更主流 |
| LR 监控指标 | `monitor='val_acc'` | `monitor='val_loss'` | 效果差别不大 |

---

## 依赖

```
torch>=2.0
torchvision
pandas
numpy
scikit-learn
matplotlib
seaborn
tqdm
```
