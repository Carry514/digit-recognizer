# Digit Recognizer

手写数字识别（MNIST）— 基于 Kaggle 教学 notebook 的课程设计，Kaggle 得分 0.99482（Top 4%）

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
│   ├── 02_model.py         # CNN 模型定义、训练与保存
│   ├── 03_evaluate.py      # 模型评估与错误分析
│   └── 04_predict.py       # Kaggle 测试集预测与提交文件生成
├── docs/                   # 文档
│   ├── tutorial_guide.md   # 教程精读笔记（中英双语）
│   ├── plan.md             # 实现计划
│   └── 2026-06-26-digit-recognizer-design.md
├── data/                   # 数据集（不上传 Git）
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── outputs/                # 输出（不上传 Git）
├── introduction-to-cnn-keras-0-997-top-6.ipynb  # 原始教程 notebook
├── pyproject.toml          # 项目配置与依赖
└── README.md
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
| 验证集准确率 | **99.52%** |
| 验证集错误率 | 0.48% |
| Kaggle 得分 | **0.99482**（Top 4%） |
| Kaggle 排名 | ~130/4000+ |

## 许可

本项目仅用于课程学习目的，原始教程版权归 Yassine Ghouzam 所有。
