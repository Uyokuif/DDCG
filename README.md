# D2C: Dual-Critic Discriminator-to-Critic Method

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 项目简介

D2C (Dual-Critic Discriminator-to-Critic) 是一个基于双重评论家架构的智能体决策系统，专为虚拟环境中的任务规划而设计。该系统结合了**可执行性评论家(CriticE)**和**质量评论家(CriticQ)**，通过分层评分机制提升智能体的决策质量。

## 🏗️ 核心架构

### 双重评论家系统
- **CriticE (Executability Critic)**: 评估动作的可执行性，确保动作在环境中能够成功执行
- **CriticQ (Quality Critic)**: 评估动作的质量，包含三个子模型（保守型、平衡型、激进型）

### 融合策略
- **disagreement_dynamic**: 基于评论家间分歧的动态权重融合
- **adaptive**: 根据动作重要性自适应选择策略
- **conservative**: TD3风格的保守策略
- **average**: 简单平均融合

## 📁 项目结构

```
d2c/
├── VirtualHome/                    # VirtualHome环境实验
│   ├── behavior_cloning/          # 智能体行为克隆
│   │   ├── interactive_interface.py  # D2C推理接口
│   │   ├── memory_graph.py       # 图记忆系统
│   │   └── utils_bc/             # 工具函数
│   └── roberta_training/         # 模型训练
│       ├── ds_train_dgap_v2_critics.py  # 训练脚本
│       └── train_dgap_v2_*.sh    # 训练启动脚本
├── Discriminator/                 # 判别器训练
│   └── VirtualHome/              # VirtualHome判别器
├── ScienceWorld/                  # ScienceWorld环境实验
└── assert/                       # 断言和验证
```

## 🚀 快速开始

### 环境要求
- Python 3.9+
- PyTorch 2.0+
- CUDA 11.8+
- DeepSpeed

### 安装依赖
```bash
pip install torch transformers deepspeed sentence-transformers
```

### 训练Critic模型
```bash
cd VirtualHome/roberta_training

# 训练CriticE (可执行性)
./train_dgap_v2_executability.sh

# 训练CriticQ (质量评估)
./train_dgap_v2_quality.sh
```

### 运行推理
```bash
cd VirtualHome/behavior_cloning
python run_d2c_evaluation.py --test_examples 10 --subset NovelScenes
```

## 📊 数据集

### 训练数据规模
- **专家数据**: 37k样本 (score=10)
- **硬负样本**: 80k样本 (score=0-2) - 语法破坏+前置条件违反
- **次优数据**: 340k样本 (score=3-9) - Flan-T5生成+相似度评分
- **总计**: 457k样本

### 数据生成
```bash
# 生成硬负样本
python virtualhome/dataset/generate_hard_negatives.py

# 生成次优数据
python virtualhome/dataset/generate_suboptimal_data.py

# 合并数据集
python virtualhome/dataset/create_final_dataset.py
```

## 🎮 环境支持

### VirtualHome
- 支持多种家庭场景
- 丰富的物体交互
- 真实物理模拟

### ScienceWorld
- 科学推理任务
- 多步骤问题解决
- 知识密集型任务

## 📈 性能指标

### 评估指标
- **Success Rate (SR)**: 任务完成率
- **Executability (EXEC.)**: 动作可执行率
- **Planning Quality**: 规划质量评分

### 实验结果
- **NovelScenes**: 随机采样10个任务进行测试
- **融合策略**: disagreement_dynamic表现最佳
- **搜索机制**: 评分<4时触发硬搜索

## 🔧 核心算法

### D2C评分流程
1. **CriticE检查**: 评估动作可执行性
2. **CriticQ评估**: 三个子模型并行评分
3. **动态融合**: 基于分歧程度调整权重
4. **最终决策**: 输出1-10分评分

### 硬搜索机制
- **触发条件**: D2C评分 < 4
- **搜索空间**: 所有可能的动作组合
- **过滤策略**: 语义+空间+状态三层过滤
- **验证机制**: 图模拟验证可执行性

## 📝 引用

如果您使用了本代码，请引用：

```bibtex
@article{d2c2024,
  title={D2C: Dual-Critic Discriminator-to-Critic Method for Task Planning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- VirtualHome团队提供的仿真环境
- Hugging Face提供的预训练模型
- DeepSpeed团队提供的分布式训练框架
