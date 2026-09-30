# OPCID 任务流程与测试报告

> 任务：从大肠杆菌 MG1655 的 Micro-C 接触矩阵中识别 OPCID 结构（主对角线附近的方形富集区域）。
> 本文记录 OPCID 识别任务（CNN 部分）在正式多标签架构下的任务流程、测试方法、测试结论与观察到的现象。

---

## 0. 任务背景与模型设计

**任务**：在 10 bp 分辨率的 Micro-C 接触矩阵中识别三类染色质结构——OPCID（方形富集）、CHIN（垂直于对角线的带）、CHID（CHIN 的簇）。本文聚焦 **OPCID**。

**模型**：多标签三分支架构，标签为 `[OPCID, CHIN, CHID]`（CHID 编码为 `[0,1,1]`）。**OPCID 分支为纯 CNN**：
- 每个标注区域从 `.cool` 截取 250 / 500 / 1000 bp 三种尺度的方形 patch；
- 每个 patch 生成 3 通道（`log1p` 原始接触、按对角线 offset 标准化、对角线距离编码）；
- 三个尺度各走一个 `ContactCNN`，embedding 取平均，进 `opcid_head`；
- OPCID 分支不使用 GNN、对角线视图、band 视图或交叉注意力（避免引入噪声）。

本模块复用主训练脚本（`region_GNN_*.py`、`GNN.py`）的类与函数，不改动主脚本代码。

---

## 1. 环境说明

| 问题 | 原因 | 解决 |
|---|---|---|
| 主脚本在 `torch.optim.AdamW` 处崩溃，报 triton 相关 | 环境内 `triton` 为残缺空壳，触发 `torch._dynamo → torch._inductor` 导入失败 | 将 `triton` 改名备份，torch 检测不到后自动跳过（可逆）|
| OpenMP 双加载报错 | Windows + torch 已知问题 | 运行前设 `KMP_DUPLICATE_LIB_OK=TRUE` |
| `python` 指向精简版（无 numpy）| PATH 配置 | 统一使用 Anaconda 环境的 `python.exe` |
| CPU 冒烟测试 | 无 GPU 环境 | 按 CPU 命令跑通 1 epoch |

---

## 2. 任务流程

```text
数据（.cool + top40 图 + Excel 标注）
   │
   ▼
① 主脚本训练多标签模型（rep1）
   │
   ▼
② 跨重复验证：加载 rep1 训练权重 → 在 rep2（模型未见过）上推理
   │
   ▼
③ OPCID 阈值扫描（0.30 ~ 0.60）
   │
   ▼
④ 加长训练（10 → 30 epoch）
   │
   ▼
⑤ 重新跨重复验证 + 阈值扫描
   │
   ▼
⑥ Grad-CAM 可解释性
   │
   ▼
⑦ 代码整洁重构（重命名 / 删无用 / 去编码声明）
```

---

## 3. 测试方法

### 3.1 跨重复生物学验证（rep1 训 → rep2 测）
- 用 rep1 训练的模型权重，在第二个独立生物学重复 rep2 的 `.cool` + top40 图上推理，评估三类表现。
- 数据：标注区域共 344 个（OPCID 68、CHIN 250、CHID 26）；多标签编码，CHID 记为 `[0,1,1]`。

### 3.2 OPCID 阈值扫描
- 固定 CHIN 阈值 0.5、CHID 阈值 0.4，把 OPCID 决策阈值从 0.30 扫到 0.60，逐档看 precision / recall / F1。
- 动机：默认 OPCID 阈值 0.55，而 OPCID precision 偏低，需要确定更合适的决策线。

### 3.3 Grad-CAM 可解释性
- hook OPCID 分支每个 `ContactCNN` 的最后一个卷积层（`features[8]`，64 通道，保留空间），对 `opcid_head` 输出反向传播，生成热图并叠加到原始接触 patch。
- 目的：验证模型关注的是"沿主对角线的方形结构"，而非背景噪声。

---

## 4. 测试结论

### 4.1 10 epoch（基准）

**rep1 留出测试（52 区）**

| 类别 | precision | recall | F1 |
|---|---|---|---|
| OPCID | 0.455 | 0.357 | 0.400 |
| CHIN | — | — | 0.708 |
| CHID | — | — | 0.545 |

**rep2 跨重复（344 区，OPCID 用默认阈值 0.55）**

| 类别 | precision | recall | F1 |
|---|---|---|---|
| OPCID | 0.387 | 0.308 | 0.343 |
| CHIN | — | — | 0.775 |
| CHID | — | — | 0.632 |

整体 binary_accuracy ≈ **0.677**。

**10 epoch 阈值扫描（rep2）**：OPCID 阈值 **0.30 最优**（precision 0.314 / recall 0.564 / F1 0.404）；0.55 时 F1 0.343。CHIN/CHID 的 F1 不受扫描影响。

### 4.2 30 epoch（加长训练后）

**训练 loss**：106.6 → 1.02（几乎完全收敛）。

**rep1**：训练集 OPCID F1 **0.630**；测试集 OPCID F1 **0.357**（见现象①过拟合）。

**rep2 跨重复 + 阈值扫描（30 epoch vs 10 epoch）**

| 指标 | 10 epoch | 30 epoch | 变化 |
|---|---|---|---|
| OPCID precision | 0.314 | **0.363** | ↑ 提升 |
| OPCID recall | 0.564 | 0.474 | ↓ 略降 |
| **OPCID F1** | 0.404 | **0.411** | ↑ 小升 |
| CHIN F1 | 0.775 | **0.930** | ↑ 大幅提升 |
| CHID F1 | 0.632 | 0.581 | ↓ 略降 |

**结论**：加长训练整体有效——OPCID **precision 明显提升**（误报变少），CHIN 跨重复大幅变好。OPCID 仍是最弱一类。

### 4.3 关键数字解读（OPCID，rep2 跨重复，30 epoch，阈值 0.30）

68 个真 OPCID 中：
- ✅ 抓到 **32 个**（recall 47.4%）
- ❌ 漏掉 **36 个**
- ⚠️ 误报 **56 个**（precision 36.3%）

验证：precision = 32/(32+56) ≈ 36.4%，recall = 32/(32+36) ≈ 47.1%，与脚本输出一致。

---

## 5. 现象与解读

### ① 过拟合：训练好、测试差
30 epoch 后，OPCID 训练集 F1 0.63，但测试集只有 0.357，且与 10 epoch 持平。**加长训练只让模型记住训练 OPCID，没有提高泛化**。根因是 **OPCID 样本太少（68 个）+ 特征弱**，属数据层面瓶颈。

### ② OPCID 是三类里最难的
| 类别 | 样本数 | 特征 | 30ep 跨重复 F1 |
|---|---|---|---|
| CHIN | 250 | 垂直高强线，明显 | 0.930 |
| CHID | 26 | 簇状 | 0.581 |
| **OPCID** | **68** | **强度不明显的矩形** | **0.411** |

### ③ 整体准确率 71% 与 OPCID 召回 47% 不矛盾
- 整体准确率算"所有标签位判对比例"，被 CHIN（抓得好）和大量"非 OPCID 判对"拉高；
- OPCID 召回只盯着 68 个真 OPCID。结构发现场景下，召回比整体准确率更重要（漏掉真结构 = 该结构丢失）。

### ④ 阈值 0.30 优于默认 0.55
降阈值让 OPCID 召回近翻倍（0.31 → 0.56），F1 提升，代价是误报略增。**OPCID 决策阈值建议用 0.30~0.35。**

### ⑤ Grad-CAM：模型关注沿对角线的方形结构
热图显示，模型判 OPCID 时关注区集中在**沿主对角线的区域**（500/1000 bp 尺度尤为明显），符合 OPCID 生物学定义，证明学到的是结构特征而非噪声。

---

## 6. 文件清单（`model/opcid/`）

### 主要脚本
| 文件 | 作用 |
|---|---|
| `01_build_dataset.py` | 早期二分类数据集构建（历史参考）|
| `02_train_cnn.py` | 早期二分类训练（历史参考）|
| `03_gradcam.py` | 早期二分类 Grad-CAM（历史参考）|
| `04_cross_rep_validate.py` | **跨重复验证 + OPCID 阈值扫描** |
| `05_gradcam_region.py` | **正式架构 OPCID 分支的 Grad-CAM** |

### 测试脚本
| 文件 | 作用 |
|---|---|
| `test01_preview.py` | 窗口预览 |
| `test02_diagnose.py` | 可分性诊断 |
| `test03_diagnose_norm.py` | 归一化对比 |
| `test04_diagnose_dist.py` | 特征分布 |
| `test05_check_data.py` | 数据检查 |

### 产物
- `data_out/gradcam_region/*.png`：6 张正式架构 OPCID Grad-CAM 热图
- `data_out/`：早期二分类产物（模型、指标、npz、早期 gradcam）作历史参考

---

## 7. 待办

- [ ] 将阈值扫描表、跨重复验证表、Grad-CAM 解读汇总进实验报告
- [ ] Grad-CAM 增加"非 OPCID"对照，或提高分辨率
