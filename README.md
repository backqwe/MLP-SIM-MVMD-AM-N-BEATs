# MLP-SIM-MVMD-AM-N-BEATS

基于深度学习的滨海河流氨氮预测框架

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
