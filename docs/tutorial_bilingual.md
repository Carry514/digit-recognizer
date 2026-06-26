# Yassine Ghouzam: Introduction to CNN Keras — Acc 0.997 (top 8%)
# 双语翻译版（英文原文 + 中文注释）

> 原教程：https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6  
> 作者：Yassine Ghouzam, PhD — 18/07/2017  
> 本文件完全按照源文件 `introduction-to-cnn-keras-0-997-top-6.ipynb` 逐 cell 对比编写。

---

## 1. Introduction / 引言

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')
```

> 导入所有需要的库。Keras 用 Sequential 搭建模型，ImageDataGenerator 做数据增强。  
> `np.random.seed(2)` 在导入区就设置随机种子，保证结果可复现。

---

## 2. Data Preparation / 数据准备

### 2.1 Load data / 加载数据

```python
# Load the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
```

> 读取训练集 (42000 行, 785 列) 和测试集 (28000 行, 784 列)。  
> 测试集没有 label 列，需要预测后提交。

```python
Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels = ["label"],axis = 1) 

# free some space
del train 

g = sns.countplot(Y_train)

Y_train.value_counts()
```

> 分离标签和特征。`del train` 释放内存（Kaggle 环境内存有限）。  
> We have similar counts for the 10 digits.  
> 标签分布：10 类数字大致均衡，每类约 4000 张，不需要做类别不平衡处理。

### 2.2 Check for null and missing values / 检查缺失值

```python
# Check the data
X_train.isnull().any().describe()
```

```python
test.isnull().any().describe()
```

> I check for corrupted images (missing values inside).  
> There is no missing values in the train and test dataset.  
> 确认数据完整，没有缺失值。

### 2.3 Normalization / 归一化

> We perform a grayscale normalization to reduce the effect of illumination's differences.  
> Moreover the CNN converg faster on [0..1] data than on [0..255].

```python
# Normalize the data
X_train = X_train / 255.0
test = test / 255.0
```

> 像素值 0-255 归一化到 0-1。CNN 对输入范围敏感，归一化有助于梯度下降更快收敛。

### 2.3 Reshape / 重塑维度（原教程标题编号为 2.3，实际应为 2.4）

> Train and test images (28px x 28px) has been stock into pandas.Dataframe as 1D vectors of 784 values.  
> We reshape all data to 28x28x1 3D matrices. Keras requires an extra dimension in the end which correspond to channels. MNIST images are gray scaled so it use only one channel.

```python
# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
```

> 将 784 个像素的一维数组 reshape 成 28×28×1 的灰度图像。  
> CNN 需要 (H, W, Channel) 格式输入，channel=1 表示灰度图。  
> （PyTorch 中需要 CHW 格式，即 (1, 28, 28)，代码中会用 permute 转换）

### 2.5 Label Encoding / 标签编码

```python
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
Y_train = to_categorical(Y_train, num_classes = 10)
```

> Labels are 10 digits numbers from 0 to 9. We need to encode these labels to one hot vectors.  
> 将整数标签 (0-9) 转为 one-hot 向量。例如标签 2 变成 [0,0,1,0,0,0,0,0,0,0]。  
> **⚠️ PyTorch 差异：CrossEntropyLoss 内置 softmax，不需要 one-hot，直接用整数标签即可。**

### 2.6 Split training and validation set / 拆分训练验证集

```python
# Set the random seed
random_seed = 2
```

```python
# Split the train and the validation set for the fitting
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
```

> I choosed to split the train set in two parts: a small fraction (10%) became the validation set which the model is evaluated and the rest (90%) is used to train the model.  
> Since we have 42,000 training images of balanced labels, a random split doesn't cause some labels to be over represented.

```python
# Some examples
g = plt.imshow(X_train[0][:,:,0])
```

> 90% 训练，10% 验证。random_seed=2 保证每次拆分结果一致。

---

## 3. CNN / 卷积神经网络

### 3.1 Define the model / 定义模型

> I used the Keras Sequential API, where you have just to add one layer at a time.  
> The first is the convolutional (Conv2D) layer. I choosed to set 32 filters for the two first Conv2D and 64 filters for the two last ones.  
> The second important layer is pooling (MaxPool2D). This layer acts as a downsampling filter.  
> Dropout is a regularization method, where a proportion of nodes are randomly ignored for each training sample.  
> The Flatten layer is use to convert the final feature maps into a one single 1D vector.  
> In the end i used two fully-connected (Dense) layers as ANN classifier. The last layer outputs probability distribution of each class.

```python
# Set the CNN model 
# my CNN architecture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
```

```
网络结构总结：
┌────────────────────────────────┬──────────────┬─────────────────────┐
│ 层                             │ 参数         │ 作用                │
├────────────────────────────────┼──────────────┼─────────────────────┤
│ Conv2D → ReLU                  │ 32, kernel=5 │ 提取低级特征（边缘） │
│ Conv2D → ReLU                  │ 32, kernel=5 │ 加深特征提取         │
│ MaxPool2D + Dropout(0.25)      │ pool=2       │ 降采样，防过拟合     │
├────────────────────────────────┼──────────────┼─────────────────────┤
│ Conv2D → ReLU                  │ 64, kernel=3 │ 提取中级特征         │
│ Conv2D → ReLU                  │ 64, kernel=3 │ 加深                 │
│ MaxPool2D + Dropout(0.25)      │ pool=2       │ 降采样               │
├────────────────────────────────┼──────────────┼─────────────────────┤
│ Flatten                        │ -            │ 展平为一维向量       │
│ Dense(256) → ReLU              │ 256 神经元   │ 全连接，综合特征     │
│ Dropout(0.5)                   │ -            │ 大幅防过拟合         │
│ Dense(10) → Softmax            │ 10 类输出    │ 输出各类概率         │
└────────────────────────────────┴──────────────┴─────────────────────┘

参数量约 200 万。Dropout(0.5) 在最后的全连接层前作用最大，因为那里参数最密集。
```

### 3.2 Set the optimizer and annealer / 设置优化器和学习率衰减

> We define the loss function to measure how poorly our model performs: "categorical_crossentropy" for >2 classes.  
> The most important function is the optimizer. I choosed RMSprop (with default values), it is a very effective optimizer.  
> The metric function "accuracy" is used to evaluate the performance. This metric is not used when training the model (only for evaluation).

```python
# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
```

```python
# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
```

> 优化器：**RMSprop**（lr=0.001, rho=0.9）。  
> 损失函数：categorical_crossentropy（对应 one-hot 标签）。  
> **⚠️ 与我们的 PyTorch 方案的差异**：教程用 RMSprop，我们用 Adam。两者性能相近。

> In order to make the optimizer converge faster, i used an annealing method of the learning rate (LR).  
> With the ReduceLROnPlateau function, i choose to reduce the LR by half if the accuracy is not improved after 3 epochs.

```python
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
```

> ReduceLROnPlateau：监控 **`val_acc`（验证准确率）**，连续 3 个 epoch 不提升时，学习率减半。  
> **⚠️ 与我们的 PyTorch 方案的差异**：教程监控 `val_acc`，我们方案监控 `val_loss`。两种方式都可以。

```python
epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86
```

> 教程中设 epochs=1 是为了在 Kaggle 上快速演示。  
> **实际要跑 30 个 epoch** 才能达到 99.67% 准确率。batch_size=86。

### 3.3 Data augmentation / 数据增强

> In order to avoid overfitting, we need to expand artificially our handwritten digit dataset.  
> The improvement is important: Without data augmentation i obtained 98.114%, with data augmentation i achieved 99.67%.

```python
# With data augmentation to prevent overfitting (accuracy 0.99286)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(X_train)
```

> 数据增强参数：旋转±10°、缩放 90%-110%、平移±10%。  
> 水平/垂直翻转设为 False——数字 6 翻转后不再是 6。

```python
# Fit the model
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])
```

> 教程使用 `model.fit_generator()`（旧版 Keras API）。`datagen.flow()` 实时产生增强后的 batch。  
> 训练时用增强数据，验证时用原始数据。

---

## 4. Evaluate the model / 模型评估

### 4.1 Training and validation curves / 训练和验证曲线

```python
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
```

> 画出 loss 和 accuracy 曲线（训练集 vs 验证集）。  
> 教程提到：validation accuracy 几乎一直高于 training accuracy，说明模型没有过拟合。

### 4.2 Confusion matrix / 混淆矩阵

```python
# Look at confusion matrix 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 
```

> Here we can see that our CNN performs very well on all digits with few errors.  
> However, it seems that our CNN has some little troubles with the 4 digits, they are misclassified as 9.  
> 混淆矩阵：对角线越亮越好。教程发现 4 容易被误判为 9。

### 4.3 Display error results / 展示错误样本

```python
# Display some error results 

# Errors are difference between predicted labels and true labels
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = Y_pred[errors]
Y_true_errors = Y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1

# Probabilities of the wrong predicted numbers
Y_pred_errors_prob = np.max(Y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, Y_pred_classes_errors, Y_true_errors)
```

> The most important errors are also the most intrigous. Some of these errors can also be made by humans.  
> 展示预测错误最严重的 6 张图。标注"预测标签 vs 真实标签"，答辩加分项。

---

## 5. Prediction and submission / 预测与提交

### 5.1 Predict and Submit results

```python
# predict results
results = model.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
```

```python
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
```

> 预测测试集 28000 张图，取概率最大的类别。  
> 生成 Kaggle 提交文件 `cnn_mnist_datagen.csv`。格式：ImageId (1-28000), Label (0-9)。

---

## Summary / 总结

Ghouzam 这个教程能拿到 **99.67% (top 8%)** 的核心要素：

| 要素 | 作用 |
|------|------|
| 双层卷积结构 | 第一层 5×5 大卷积核捕捉大特征，第二层 3×3 细化 |
| Dropout (0.25 → 0.25 → 0.5) | 逐层增强正则化，全连接层前最激进 |
| 数据增强 | 旋转+平移+缩放，准确率从 98.1% 提升到 99.67% |
| ReduceLROnPlateau(monitor='val_acc') | 监控验证准确率，自适应衰减学习率 |
| RMSprop 优化器 | 自适应学习率，比普通 SGD 好调 |

---

## PyTorch 翻译对照表

| Keras | PyTorch |
|-------|---------|
| `Sequential()` | `nn.Sequential()` 或 `nn.Module` |
| `Conv2D(32, 5, padding='Same')` | `nn.Conv2d(1, 32, 5, padding=2)` |
| `MaxPool2D(2,2)` | `nn.MaxPool2d(2)` |
| `Dropout(0.25)` | `nn.Dropout(0.25)` |
| `Flatten()` | `nn.Flatten()` 或 `x.view(x.size(0), -1)` |
| `Dense(256, 'relu')` | `nn.Linear(3136, 256)` + `F.relu()` |
| `Dense(10, 'softmax')` | `nn.Linear(256, 10)`（CrossEntropyLoss 自带 softmax）|
| `ImageDataGenerator(...)` | `transforms.RandomAffine(...)` |
| `to_categorical(y, 10)` | 无需转换，直接用整数标签 |
| `model.fit_generator(...)` | 手写训练循环 |
| `ReduceLROnPlateau(monitor='val_acc')` | `ReduceLROnPlateau(monitor='val_loss')` **←差异** |
| `RMSprop(lr=0.001)` | `Adam(lr=0.001)` **←差异** |
