# Digit Recognizer

手写数字识别（MNIST）— 基于 Kaggle 教学 notebook 的课程设计，Kaggle 得分 0.99482（前 16%）

## 项目简介

本课程设计学习并复现了 Kaggle 经典教程 [Introduction to CNN Keras - 0.997 (top 8%)](https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)（作者：Yassine Ghouzam, PhD），将模型实现从 Keras 翻译为 PyTorch，在 MNIST 手写数字数据集上达到了 **99.5%+** 的验证准确率。

## 参考来源

- **原始教程**：[Introduction to CNN Keras - Acc 0.997 (top 8%)](https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)
- **作者**：Yassine Ghouzam, PhD
- **平台**：Kaggle Digit Recognizer Competition

## 技术栈

- **语言**：Python 3.13
- **框架**：PyTorch（原教程使用 Keras，本项目翻译为 PyTorch）
- **模型**：卷积神经网络（CNN）
- **数据**：MNIST（28×28 灰度手写数字）
- **依赖管理**：uv（pyproject.toml）

## 项目结构

```
digit-recognizer/
├── src/                    # 源代码
│   ├── 01_data_prep.py     # 数据加载与预处理
│   ├── 02_model.py         # CNN 训练（val_acc 最优保存）
│   ├── 03_evaluate.py      # 混淆矩阵 + 错误分析
│   ├── 04_predict.py       # 预测 + 生成 submission.csv
│   ├── 01_data_prep_cv.py  # CV 版数据准备（导出 X_full）
│   ├── 02b_model_acc.py    # acc 版训练（patience=8）
│   ├── 02_cv_train.py      # 3 种子交叉验证训练
│   ├── 03b_evaluate_acc.py # acc 版评估
│   ├── 04b_predict_acc.py  # acc 版预测
│   ├── 04_ensemble.py      # 2 模型集成预测
│   └── 04_cv_ensemble.py   # CV 3 模型集成预测
├── docs/                   # 文档
├── data/                   # 数据集（Git 忽略）
├── outputs/                # 产出（Git 忽略）
│   ├── model_best.pth      # 最优模型权重
│   ├── training_curves.png # 训练曲线
│   ├── confusion_matrix.png
│   ├── error_samples.png
│   ├── submission.csv      # Kaggle 提交文件
│   └── ...
└── report/                 # 报告生成脚本（Git 忽略）
```

## 快速开始

```bash
# 1. 安装依赖
uv sync

# 2. 准备数据：将 train.csv 和 test.csv 放入 data/ 目录
#    （从 Kaggle Digit Recognizer 下载：https://www.kaggle.com/c/digit-recognizer/data）

# 3. 数据预处理
uv run python src/01_data_prep.py

# 4. 训练模型
uv run python src/02_model.py

# 5. 评估结果
uv run python src/03_evaluate.py

# 6. 生成 Kaggle 提交文件
uv run python src/04_predict.py
```

## 结果

| 指标 | 数值 |
|------|------|
| 训练最高 val_acc | **99.64%** |
| 验证集准确率 | 99.52% |
| 验证集错误率 | 0.48% (20/4200) |
| **Kaggle 得分** | **0.99482** |
| Kaggle 排名 | 249 / 1533 (前 16%) |

### 优化方案对比

| 方案 | Kaggle 得分 | 说明 |
|------|------------|------|
| **原始单模型** | **0.99482** | 基准线，最高 |
| TTA 测试时增强 | 0.99439 | 引入噪声 |
| Acc 保存版 | 0.99450 | 不如原始 |
| 2 模型 Ensemble | 0.99467 | 错误重叠 |
| 3 模型 CV Ensemble | 0.99457 | 互补但未超基线 |

> 所有优化方案均未能超越原始单模型。0.99482 可能是当前 CNN 架构的上限。

## 许可

本项目仅用于课程学习目的，原始教程版权归 Yassine Ghouzam 所有。
