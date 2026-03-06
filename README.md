# MLP-SIM-MVMD-AM-N-BEATs

基于深度学习的时间序列预测框架，结合 MLP-SIM-MVMD 与 AM-N-BEATs 技术

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 项目简介

本仓库实现了一种新型两阶段因果蒸馏框架，用于氨氮（NH₃-N）预测，结合了以下技术：

- **MLP** (Multi-Layer Perceptron) — 多层感知器
- **SIM** (Similarity Module) — 相似度模块
- **MVMD** (Multivariate Variational Mode Decomposition) — 多元变分模态分解
- **AM** (Attention Mechanism) — 注意力机制
- **N-BEATs** (Neural Basis Expansion Analysis for Time Series) — 时间序列神经基扩展分析

## 技术栈

- Python 3.8+
- PyTorch 1.13+
- NumPy, Pandas, SciPy, Matplotlib, Scikit-learn

## 安装方法

```bash
# 克隆仓库
git clone https://github.com/backqwe/MLP-SIM-MVMD-AM-N-BEATs.git
cd MLP-SIM-MVMD-AM-N-BEATs

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 可选：以可编辑模式安装包
pip install -e .
```

## 使用方法

### 训练模型

```bash
python train.py --config config/default_config.yaml
```

### 预测/推理

```bash
python predict.py --config config/default_config.yaml --checkpoint checkpoints/best_model.pth
```

## 项目结构

```
MLP-SIM-MVMD-AM-N-BEATs/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
├── config/
│   └── default_config.yaml    # 默认配置文件
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py             # MLP 模块
│   │   ├── nbeats.py          # N-BEATs 模型
│   │   ├── attention.py       # 注意力机制模块
│   │   ├── mvmd.py            # MVMD 模块
│   │   ├── sim_module.py      # SIM 模块
│   │   └── ensemble.py        # 组合模型管线
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py         # 数据集加载与预处理
│   │   └── preprocessing.py   # 数据预处理工具函数
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py         # 评估指标
│       ├── visualization.py   # 可视化工具
│       └── logger.py          # 日志工具
├── train.py                   # 训练入口脚本
├── predict.py                 # 预测/推理脚本
└── tests/
    ├── __init__.py
    ├── test_models.py          # 模型单元测试
    └── test_data.py            # 数据处理单元测试
```

## 数据集

本仓库包含了论文中使用的多组数据集，用于结果复现。这些数据集包含了中国大连大沙河（2021-2024）的水质监测数据。

---

# MLP-SIM-MVMD-AM-N-BEATs

A deep learning framework for time series prediction combining MLP-SIM-MVMD and AM-N-BEATs.

## Description

This repository implements a novel two-stage causal distillation framework for NH₃-N prediction, combining:

- **MLP** (Multi-Layer Perceptron)
- **SIM** (Similarity Module)
- **MVMD** (Multivariate Variational Mode Decomposition)
- **AM** (Attention Mechanism)
- **N-BEATs** (Neural Basis Expansion Analysis for Time Series)

## Installation

```bash
git clone https://github.com/backqwe/MLP-SIM-MVMD-AM-N-BEATs.git
cd MLP-SIM-MVMD-AM-N-BEATs
pip install -r requirements.txt
```

## Quick Start

```bash
# Train
python train.py --config config/default_config.yaml

# Predict
python predict.py --config config/default_config.yaml --checkpoint checkpoints/best_model.pth
```

## Dataset

The repository includes multiple datasets from the Dasha River, Dalian, China (2021-2024) for result reproduction.

## C++ Implementation

A C++ implementation using LibTorch is also available. See the `main.cpp` and `CMakeLists.txt` files for details.

### Prerequisites for C++ build

- C++ 17 or higher
- LibTorch (PyTorch C++ API)
- CMake 3.14 or higher

## 项目简介

本仓库包含了一种新型两阶段因果蒸馏框架的实现，用于氨氮（NH₃-N）预测，结合了 MLP-SIM-MVMD（多层感知器模拟多元变分模态分解）和 AM-N-BEATS（注意力机制增强的 N-BEATS）架构。

## 安装与使用

### 环境要求

- C++ 17 或更高版本
- LibTorch（PyTorch C++ API）
- CMake 3.14 或更高版本

### 配置步骤

1. **下载源代码**
2. **解压源代码**
   如果下载的是压缩包，请将所有文件解压到工作目录。
3. **下载并配置 LibTorch**
   从 [PyTorch 官网](https://pytorch.org/) 下载适用于您系统的 LibTorch 包，将 LibTorch 包解压到项目根目录
   确保目录结构如下：
   
             项目根目录/
          ├── main.cpp
          ├── resources/
          ├── libtorch/          # LibTorch 解压到此处
          └── CMakeLists.txt
4. **环境配置**
   配置外部链接 使用 CMake 或其他方式正确链接 LibTorch 库，确保所有依赖项被正确引用。
6. **编译并运行**

### 数据集

本仓库包含了论文中使用的多组数据集，用于结果复现。这些数据集包含了中国大连大沙河（2021-2024）的水质监测数据。


# ---------------------------------------------------------------

# MLP-SIM-MVMD-AM-N-BEATS

A deep learning-based ammonia nitrogen (NH₃-N) prediction framework for coastal river water quality monitoring.

## Description

This repository contains the implementation of a novel two-stage causal distillation framework for NH₃-N prediction, combining MLP-SIM-MVMD (Multi-Layer Perceptron for Simulating Multivariate Variational Mode Decomposition) and AM-N-BEATS (Attention Mechanism enhanced N-BEATS) architectures.

## Installation and Usage

### Prerequisites

- C++ 17
- LibTorch (PyTorch C++ API)
- CMake 3.14 or higher

### Configuration Steps

1. **Download the source code**
2. **Extract the source code**
   If downloaded as a compressed package, extract all files to your working directory.
3. **Download and configure LibTorch**
   Download the appropriate LibTorch package from the [official PyTorch website](https://pytorch.org/)
   Extract the LibTorch package to the root directory of this project
   Ensure the directory structure is as follows:

           project-root/
        ├── main.cpp
        ├── resources/
        ├── libtorch/          # LibTorch package extracted here
        └── CMakeLists.txt

4. **Environment Configuration**
   Configure external linking Configure CMake or your preferred build system to properly link the LibTorch library. Ensure all dependencies are correctly referenced.
5. **Build and run**

### Dataset

The repository includes multiple datasets used in the research paper for result reproduction. These datasets contain water quality monitoring data from the Dasha River, Dalian, China (2021-2024).
