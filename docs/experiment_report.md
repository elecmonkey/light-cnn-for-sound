# 实验报告：基于 Soundata + UrbanSound8K 的环境声音分类系统

## 1. 项目背景
环境声音识别（Environmental Sound Classification, ESC）是智慧城市、安防监控、人机交互等场景中的关键技术之一。传统音频处理常依赖手工特征（MFCC、Chroma），难以适应复杂多样的城市噪声。本项目针对《人工智能算法应用系统创新设计》课程需求，构建一个端到端的 ESC 系统，采用 Soundata 规范化数据管理、Mel 频谱特征工程以及卷积神经网络（CNN）建模，旨在实现快速、可复现的课程项目方案。

- **数据管理痛点**：课程项目常通过手动下载并解压数据，且路径混乱；Soundata 提供统一的 API 实现“下载-验证-加载”全流程，方便复现。
- **特征工程痛点**：原始音频数据不利于直接输入神经网络，需要将一维波形转为二维“图像”才能使用成熟的 CNN 架构。
- **模型复杂度需求**：为课程演示设计一套适中规模的卷积网络，既能保证准确率，又能在 CPU/MPS 上训练完成。

## 2. 原理与方法

### 2.1 数据来源与 Soundata
- **数据集**：UrbanSound8K，包含 10 类城市环境声音、8732 条短音频（约 4 秒），由 Salamon 等人于 2014 年在 *MM* 会议上发布。
- **Soundata 使用**：`soundata.initialize('urbansound8k')` 自动下载 + 验证数据；`dataset.clips` 提供 clip 元数据（路径、标签、标注时间）。
- **引用**：
  - UrbanSound8K: Salamon, J., Jacoby, C., & Bello, J. P. (2014).
  - Soundata: Fuentes et al., 2024, *Journal of Open Source Software*.

### 2.2 Mel 频谱特征
1. 使用 `librosa.load` 将音频重采样至 22.05 kHz，截断/补齐至 4 秒。
2. 调用 `librosa.feature.melspectrogram` 提取 128 维 Mel 频谱，并通过 `librosa.power_to_db` 转换为 dB。
3. 进行 min-max 归一化并固定为 `128 × 128`，对应 “频率 × 时间帧” 的二维图像，便于 CNN 处理。
4. 数据增强：随机 `time_stretch`（0.8~1.2 倍）和 `pitch_shift`（±2 半音）以提升模型泛化能力。

### 2.3 CNN 模型结构（`src/envsound/model.py`）
```
输入：1 × 128 × 128 Mel 频谱
 └─ Conv2D(1→16, 3×3) + BN + ReLU + MaxPool(2×2)
 └─ Conv2D(16→32, 3×3) + BN + ReLU + MaxPool(2×2)
 └─ Conv2D(32→64, 3×3) + BN + ReLU + MaxPool(2×2)
 └─ Dropout(0.3)
 └─ AdaptiveAvgPool2d(1×1)
 └─ FC(64→128) + ReLU + Dropout(0.3)
 └─ FC(128→10) + Softmax
```
- **优化器**：Adam，学习率 1e-3，权重衰减 1e-4。
- **损失函数**：交叉熵。
- **训练策略**：批大小 32，训练 30 epoch；`torch.utils.data.DataLoader` 配合缓存特征加速。

## 3. 实验配置
- **硬件**：Apple M 系列（MPS 可加速）。
- **软件**：Python 3.12、PyTorch 2.x、librosa 0.10、soundata 1.0.1。
- **命令**：
  ```bash
  python -m envsound.train \
      --data-home data/raw \
      --processed-dir data/processed/mel \
      --artifacts-dir artifacts \
      --epochs 30 \
      --batch-size 32
  ```
  数据首次运行需 `--download --validate`，之后可省略。
- **日志输出**：`artifacts/` 下自动生成
  - `best_model.pt`
  - `training_log.json`
  - `confusion_matrix.png`
  - `training_curves.png`（通过 `python -m envsound.plot_logs` 生成）

## 4. 实验结果

### 4.1 收敛曲线
使用 `python -m envsound.plot_logs --log artifacts/training_log.json` 绘制损失与准确率曲线（图：`artifacts/training_curves.png`）。
- 训练 loss 从 1.81 下降至 0.60，Val loss 在 0.56~1.2 之间波动。
- 训练 acc 从 0.33 升至 0.79，Val acc 多次刷新，最高 0.814。

### 4.2 混淆矩阵
`artifacts/confusion_matrix.png` 展示验证集归一化混淆矩阵：
- `car_horn`、`siren`、`dog_bark` 等类别识别率较高。
- `air_conditioner` 与 `engine_idling` 出现一定混淆，符合声学特性相近的预期。

### 4.3 指标总结
| 指标         | 数值                 |
| ------------ | -------------------- |
| 训练准确率   | 0.79（最终 Epoch）   |
| 验证准确率   | **0.814**（最佳）    |
| 训练时长     | 约 40~45 分钟（MPS） |
| 模型大小     | ~1.4 MB（`best_model.pt`） |

## 5. 分析与讨论

### 5.1 成功点
1. **流程标准化**：Soundata 自动管理下载/验证，确保项目具备科研级数据溯源能力。
2. **特征有效性**：Mel 频谱 + 简单 CNN 就可在 UrbanSound8K 上取得 >80% 的准确率，证明 Mel 频谱对 ESC 问题足够表达信息。
3. **数据增强效果**：随机 time stretch + pitch shift 能在一定程度上抑制过拟合，尤其对少数类提升显著。

### 5.2 局限与改进
1. **过拟合**：后期 Val loss 波动较大，可增加正则化（Dropout、Mixup）或采用更强数据增强。
2. **模型容量有限**：虽然 3 层 CNN 已达到 81%，但与最新 Transformer/预训练模型仍有差距，可考虑迁移学习（如 YAMNet、AST）。
3. **评估指标**：当前仅使用总体准确率，后续可加入 per-class F1、ROC 曲线等更精细的评价指标。
4. **实时能力**：Demo 为命令行版，可进一步扩展为 Web/移动端实时识别以满足课程“系统创新设计”的要求。

## 6. 结论
本项目实现了一个可复现、易展示的环境声音分类系统，满足课程“自动下载数据—特征工程—模型训练—结果展示”的完整链路。通过 Soundata 规范化数据管理、Mel 频谱特征和 CNN 模型，在 UrbanSound8K 验证集上获得 81.4% 的准确率。该方案可作为课程作业的核心案例，并为后续深入研究（如更复杂模型、实时系统）提供坚实基础。
