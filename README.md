# AI牙科影像检测

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于深度学习的牙科图像分类和检测系统。该项目使用先进的图像处理技术来检测和分类各种牙科疾病，包括口腔病变、蛀牙、填充物和牙菌斑。

## 🎯 项目概述

本项目包含三个主要模块：

1. **口腔病变分类**：使用CNN进行8种口腔病变的图像分类
2. **口腔疾病检测**：使用深度学习进行牙科疾病的目标检测  
3. **语义分割**：使用UNet进行口腔疾病的精确分割

### ✨ 主要特性

- **高精度分类**：口腔病变分类准确率达94%+
- **实时检测**：快速的牙科疾病目标检测
- **语义分割**：精确的疾病区域分割
- **易于使用**：简洁的命令行接口
- **模块化设计**：清晰的代码架构

## 🏗️ 项目结构

```
├── train/                    # 训练脚本
│   ├── main_train.py        # 主训练入口
│   ├── train_oral_lesions.py # 口腔病变分类训练
│   ├── train_oral_conditions.py # 口腔疾病检测训练
│   ├── evaluate_model.py    # 模型评估
│   └── inference.py         # 模型推理
├── models/                  # 模型定义
│   └── neural_networks.py  # 神经网络架构
├── utils/                   # 工具函数
│   └── data_preprocessing.py # 数据预处理
├── Data/                    # 数据集目录
│   ├── oral_lesions_raw/   # 口腔病变数据
│   ├── teeth_raw/          # 牙齿图像数据
│   └── annotations/        # 标注文件
├── config.py               # 配置文件
└── requirements.txt        # 依赖包
```

## 🔧 环境要求

- Python 3.8+
- TensorFlow 2.8+
- OpenCV 4.5+
- 其他依赖见 `requirements.txt`

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/MybcyQzqxw/Dental-Imaging-Modeling-AI.git
cd Dental-Imaging-Modeling-AI

# 创建conda环境
conda create -n dental_imaging python=3.8 -y
conda activate dental_imaging

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

将数据按以下结构放置：

```
Data/
├── oral_lesions_raw/        # 口腔病变分类数据
│   ├── Gingivits/          # 牙龈炎
│   ├── Cold_Sores/         # 唇疱疹
│   ├── Canker_Sores/       # 口疮
│   ├── Periodontitis/      # 牙周炎
│   ├── Receding_Gum/       # 牙龈萎缩
│   ├── abfraction/         # 楔状缺损
│   ├── Thrush/             # 鹅口疮
│   └── Gingival_Cyst/      # 牙龈囊肿
├── teeth_raw/              # 牙齿原始图像
└── annotations/            # 检测标注文件
```

### 3. 模型训练

#### 🔸 口腔病变分类

```bash
# 基础训练
python train/main_train.py lesions --data_path ./Data

# 详细参数训练
python train/train_oral_lesions.py \
    --data_path ./Data/oral_lesions_raw \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --image_size 90 \
    --save_model my_lesions_model.h5 \
    --visualize
```

#### 🔸 口腔疾病检测

```bash
# 基础训练
python train/main_train.py conditions --data_path ./Data

# 详细参数训练
python train/train_oral_conditions.py \
    --images_path "./Data/teeth_raw/*.JPG" \
    --annotations_path "./Data/annotations/" \
    --epochs 100 \
    --batch_size 3 \
    --learning_rate 0.0001 \
    --image_size 480 \
    --save_model my_detector_model.h5
```

#### 🔸 同时训练两个模型

```bash
python train/main_train.py both --data_path ./Data
```

### 4. 模型评估

```bash
python train/evaluate_model.py \
    --model_path my_lesions_model.h5 \
    --test_data_path ./Data/oral_lesions_raw \
    --model_type classification
```

### 5. 模型推理

```bash
python train/inference.py \
    --model_path my_lesions_model.h5 \
    --image_path test_image.jpg \
    --visualize
```

## 📊 模型性能

| 模型 | 任务 | 性能指标 |
|------|------|----------|
| CNN | 口腔病变分类 | 准确率 ~94% |
| 深度检测网络 | 目标检测 | 良好的精确度和召回率 |
| UNet | 语义分割 | 受限于数据集大小 |

## 🔬 支持的疾病类型

### 口腔病变分类 (8类)
- 牙龈炎 (Gingivits)
- 唇疱疹 (Cold_Sores)
- 口疮 (Canker_Sores)
- 牙周炎 (Periodontitis)
- 牙龈萎缩 (Receding_Gum)
- 楔状缺损 (Abfraction)
- 鹅口疮 (Thrush)
- 牙龈囊肿 (Gingival_Cyst)

### 口腔疾病检测
- 牙菌斑检测
- 蛀牙检测
- 填充物识别

## 📝 训练输出

训练完成后会生成：
- `*.h5` - 训练好的模型文件
- `*_history.png` - 训练历史图表
- `*_report.txt` - 详细训练报告
- `*_confusion_matrix.png` - 混淆矩阵（分类任务）
- `predictions/` - 预测结果可视化

## ⚙️ 高级配置

可以修改 `config.py` 文件来调整：
- 数据路径
- 模型参数
- 训练超参数
- 输出路径

## 🐛 常见问题

### Q: 内存不足怎么办？
A: 减小 `batch_size` 参数，或使用更小的 `image_size`

### Q: 训练速度太慢？
A: 使用GPU训练，或减少训练数据量进行测试

### Q: 准确率不够高？
A: 增加训练轮数、启用数据增强、调整学习率

### Q: 数据加载失败？
A: 检查数据路径是否正确，确保图像格式支持

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢所有为牙科图像数据集做出贡献的研究者
- 感谢开源社区提供的优秀工具和框架
- 特别感谢深度学习社区的技术支持

## 📬 联系方式

- 项目链接：[https://github.com/MybcyQzqxw/Dental-Imaging-Modeling-AI](https://github.com/MybcyQzqxw/Dental-Imaging-Modeling-AI)
- 问题反馈：通过 GitHub Issues

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！
