# Hi-C 区域结构识别：GNN + CNN

本项目用于从 Hi-C 接触矩阵中识别三类区域结构：

- **OPCID**：主要关注原始接触矩阵中的方形区域、对称性和区域边界。
- **CHIN**：主要关注垂直于主对角线的高强度结构及其距离特征。
- **CHID**：主要关注多个 CHIN-like 区域形成的簇及其局部拓扑关系。

模型采用多标签输出。类别顺序固定为：

```text
[OPCID, CHIN, CHID]
```

CHID 是 CHIN 的簇，因此 CHID 样本通常编码为 `[0, 1, 1]`，而不是与 CHIN 互斥的单一类别。

## 项目流程

```text
.cool 接触矩阵
      |
      +--> graph_parse.py --topk K --> 稀疏 Top-K 图 .npz
      |
      +--> region_GNN_optimized_none_negative.py
              |
              +--> Excel 标签区域 --> 局部图 + 局部矩阵 patch
              |
              +--> OPCID：原始方形 CNN
              +--> CHIN：对角线/band CNN + GNN
              +--> CHID：CHIN/GNN 特征 + 簇级特征
              |
              +--> 三个独立分类 head
              +--> [p_opcid, p_chin, p_chid]
              |
              +--> 可选全基因组扫描和新候选区域输出
```

训练阶段只使用 `*_data.xlsx` 中的标注区域；全基因组滑动扫描只在加入 `--scan` 后执行。

## 数据组织

推荐目录结构：

```text
data/
  datasets/
    *.cool
    *_data.xlsx
    graph_parse.py
  graphs/
    *.npz
  region_embeddings/
model/
  GNN.py
  region_GNN.py
  region_GNN_optimized.py
  region_GNN_optimized_none_negative.py
```

`*_data.xlsx` 中应包含区域的起始坐标、结束坐标和类别信息。当前脚本读取这些标注作为监督样本，不读取旧的 sheet 4、5、6。

## 1. 构建 Top-K 图

`data/datasets/graph_parse.py` 从 `.cool` 文件读取接触矩阵，为每个 bin 保留接触强度最高的 K 个非自身邻居，并构建对称稀疏图。

例如：

```bash
python data/datasets/graph_parse.py --input data/datasets/GSE272159_37C_rep1.mapq_30.10.cool --output-dir data/graphs --topk 40
```

批量处理目录：

```bash
python data/datasets/graph_parse.py --input data/datasets --output-dir data/graphs --topk 40
```

主要参数：

| 参数 | 说明 |
|---|---|
| `--input` | `.cool` 文件或包含 `.cool` 的目录 |
| `--output-dir` | 输出 `.npz` 目录 |
| `--topk` | 每个 bin 保留的最强非自身接触数 |
| `--chunk-size` | 分块读取矩阵的行数 |
| `--min-count` | 过滤低于该接触计数的边 |

生成的图 `.npz` 主要包含：

```text
x             节点特征
edge_index    [2, E] 图边索引
edge_weight   [E] 原始 Cooler 接触强度
node_start    节点基因组起点
node_end      节点基因组终点
node_chrom    节点染色体
metadata      图构建参数和字段说明
```

节点特征包括相对染色体位置、总接触强度和 Top-K 接触强度等归一化特征。图文件不包含监督标签，标签由区域训练脚本从 Excel 读取。

## 2. 当前模型架构

### OPCID 分支

对每个标签区域，从 `.cool` 截取 250、500、1000 bp 三种上下文的方形矩阵。每个 patch 生成三个通道：

1. `log1p(raw contact)`；
2. 按距离 offset 标准化的 contact；
3. 到主对角线的距离编码。

OPCID 只使用每个尺度的原始方形视图，分别经过 CNN 后取平均，再进入 `opcid_head`。该分支不使用 GNN、对角线变换、band view 或 cross-attention。

### CHIN 分支

CHIN 使用局部图的 GNN node embedding，以及 250/500/1000 bp patch 的对角线坐标视图和 band representation。CNN token 与 GNN node embedding 通过 cross-attention 交互，并拼接区域接触统计特征后进入 `chin_head`。

### CHID 分支

CHID 在 CHIN 分支基础上加入簇级特征，例如高强度 band 比例、行列聚集程度和高强度区域质量，随后进入独立的 `chid_head`。这使 CHID head 能够关注“多个 CHIN-like 结构是否形成簇”。

### 分类和软约束

三个 head 分别输出一个 logit，经 sigmoid 后得到三个概率。训练使用带类别权重的多标签 BCE：

```text
classification_loss = BCEWithLogits([opcid, chin, chid], labels)
```

同时使用 CHID-CHIN 层级软约束：当 `p_chid > p_chin` 时增加较小惩罚。该约束不是硬规则，避免标签重叠或标注误差被放大。

## 3. 训练

完整 GPU 命令：

```bash
python model/region_GNN_optimized_none_negative.py --input data/graphs/GSE272159_37C_rep1.mapq_30.10_top40.npz --cool-input data/datasets/GSE272159_37C_rep1.mapq_30.10.cool --labels data/datasets --output-dir data/region_embeddings --epochs 30 --batch-size 32 --num-workers 16 --prefetch-factor 2 --include-background-negatives --pin-memory --persistent-workers --patch-size 64 --device cuda
```

不加入随机背景负样本时，删除：

```bash
--include-background-negatives
```

只使用 Excel 标注区域的命令：

```bash
python model/region_GNN_optimized_none_negative.py --input data/graphs/GSE272159_37C_rep1.mapq_30.10_top40.npz --cool-input data/datasets/GSE272159_37C_rep1.mapq_30.10.cool --labels data/datasets --epochs 30 --batch-size 32 --num-workers 16 --patch-size 64 --device cuda
```

CPU 调试时可使用：

```bash
python model/region_GNN_optimized_none_negative.py --input data/graphs/GSE272159_37C_rep1.mapq_30.10_top40.npz --cool-input data/datasets/GSE272159_37C_rep1.mapq_30.10.cool --labels data/datasets --epochs 1 --batch-size 2 --num-workers 0 --no-pin-memory --no-persistent-workers --device cpu
```

训练采用随机划分，默认 `--test-fraction 0.15`。`--batch-size` 是 DataLoader 的样本批大小；由于每个区域图节点数不同，batch 内部仍按区域逐样本完成图和 patch 的 forward。

### 常用训练参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--epochs` | 30 | 训练轮数 |
| `--batch-size` | 32 | 区域 batch 大小 |
| `--hidden-dim` | 32 | GNN 隐藏维度 |
| `--embedding-dim` | 16 | GNN/CNN embedding 维度，需为 4 的倍数 |
| `--classifier-hidden-dim` | 32 | 各类别 head 隐藏维度 |
| `--patch-size` | 32 | CNN patch 边长 |
| `--lr` | 0.001 | AdamW 学习率 |
| `--test-fraction` | 0.15 | 测试区域比例 |
| `--seed` | 42 | 随机种子 |
| `--num-workers` | 4 | CPU 数据加载 worker 数 |
| `--prefetch-factor` | 2 | 每个 worker 预取 batch 数 |
| `--pin-memory` | 开启 | GPU 数据传输优化 |
| `--persistent-workers` | 开启 | epoch 间保留 worker |

## 4. 预测阈值和新类型扫描

三个类别使用独立阈值：

```bash
--opcid-threshold 0.55
--chin-threshold 0.50
--chid-threshold 0.40
```

训练后进行全图扫描：

```bash
python model/region_GNN_optimized_none_negative.py --input data/graphs/GSE272159_37C_rep1.mapq_30.10_top40.npz --cool-input data/datasets/GSE272159_37C_rep1.mapq_30.10.cool --labels data/datasets --output-dir data/region_embeddings --epochs 30 --batch-size 32 --num-workers 16 --patch-size 64 --device cuda --scan --scan-window 0 --scan-step 500 --novelty-threshold 0.55
```

扫描不是把整张矩阵一次输入 CNN，而是沿基因组滑动，每次对一个局部窗口构建图和多尺度矩阵 patch。`--scan-window 0` 时使用正样本长度的中位数。

候选新区域的当前判断为：

```text
窗口不与已知标签区域重叠
且 max(p_opcid, p_chin, p_chid) < novelty_threshold
```

候选区域后续可用扫描窗口 embedding 进行 UMAP/t-SNE 和聚类；当前扫描文件主要保存坐标、三个类别概率和新候选标记。

## 5. 输出文件

默认输出目录为 `data/region_embeddings`。

### `<stem>_region_model.pt`

保存当前专用多分支模型的权重和训练参数：

```python
{
    "specialized_model": model.state_dict(),
    "args": vars(args),
}
```

该 checkpoint 包含 GNN、OPCID CNN、CHIN/CHID CNN、cross-attention 和三个独立 head。旧版单一 `classifier` 结构的 checkpoint 不能直接加载到当前架构。

### `<stem>_region_test.npz`

包含：

```text
embedding   测试区域融合表示
labels      [OPCID, CHIN, CHID] 多标签
start/end   测试区域坐标
metadata    JSON 格式训练统计和配置
```

### `<stem>_whole_genome_scan.npy`

仅加入 `--scan` 时生成，字段为：

```text
start, end
p_opcid, p_chin, p_chid
novelty_score
novel_candidate
```

## 6. 指标解释

指标数组顺序始终是 `[OPCID, CHIN, CHID]`：

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

`binary_accuracy` 是三个标签位逐元素计算的准确率，不是一个区域三个类别全部正确的比例。样本不均衡时，优先观察测试集每个类别的 precision、recall 和 F1，尤其是 OPCID。

## 7. 后台运行

```bash
nohup python model/region_GNN_optimized_none_negative.py --input data/graphs/GSE272159_37C_rep1.mapq_30.10_top40.npz --cool-input data/datasets/GSE272159_37C_rep1.mapq_30.10.cool --labels data/datasets --output-dir data/region_embeddings --epochs 30 --batch-size 32 --num-workers 16 --prefetch-factor 2 --include-background-negatives --pin-memory --persistent-workers --patch-size 64 --device cuda > region_train.log 2>&1 &
```

查看日志：

```bash
tail -f region_train.log
```

## 8. 常见问题

### `Torch not compiled with CUDA enabled`

当前 PyTorch 不支持 CUDA 时使用 `--device cpu`。如果需要 GPU，应在对应 CUDA 环境中安装匹配的 PyTorch。

### `expected 4D input`

CNN 输入必须是 `[batch, channels, height, width]`。当前 OPCID 分支使用 `[1, 3, patch_size, patch_size]`，不要手动删除 batch 维。

### 显存或速度问题

优先降低 `--batch-size`、`--patch-size` 或 `--num-workers`。`--pin-memory` 和 `--persistent-workers` 主要优化数据准备，不会降低单个模型 forward 的显存占用。

### 旧模型无法加载

当前架构有三个独立 head，必须使用当前脚本重新训练生成新的 `_region_model.pt`，不能直接加载旧版单分类器 checkpoint。

