# region_GNN_optimized_none_negative 模型架构与运行模板

## 功能

脚本读取 Top-K 图 `.npz`、原始 `.cool` 和 `*_data.xlsx` 标签，训练区域级 OPCID/CHIN/CHID 多标签模型。训练样本来自 Excel 标注区域；只有加入 `--include-background-negatives` 才会生成随机背景负样本。加入 `--scan` 后才进行全基因组滑动窗口扫描。

## 模型结构

输出顺序固定为 `[OPCID, CHIN, CHID]`，三个类别使用独立 sigmoid head，不使用互斥 softmax。

### OPCID 分支

三个尺度（250/500/1000 bp）的原始方形矩阵分别进入独立 `ContactCNN`，embedding 取平均后进入 `opcid_head`。该分支不使用 GNN、对角线视图、band 视图或 cross-attention。输入形状为 `[1, 3, patch_size, patch_size]`，三个通道是 log1p contact、距离 offset 标准化 contact 和主对角线距离编码。

### CHIN 分支

使用 GNN node embedding、对角线坐标视图、band representation、CNN-GNN cross-attention 和区域接触统计特征，输出到 `chin_head`。

### CHID 分支

继承 CHIN 的 GNN/CNN 特征，并增加每个尺度的高强度 band 比例、行列聚集程度等簇级特征，输出到 `chid_head`。CHID 标签编码为 `[0, 1, 1]`，表示 CHID 同时属于 CHIN。

## 损失

分类使用带类别权重的 `binary_cross_entropy_with_logits`。另有 CHID-CHIN 软约束：

```python
hierarchy_loss = F.relu(p_chid - p_chin).pow(2)
total_loss = classification_loss + 0.10 * hierarchy_loss
```

## 训练命令

```bash
python model/region_GNN_optimized_none_negative.py --input data/graphs/GSE272159_37C_rep1.mapq_30.10_top40.npz --cool-input data/datasets/GSE272159_37C_rep1.mapq_30.10.cool --labels data/datasets --output-dir data/region_embeddings --epochs 30 --batch-size 32 --num-workers 16 --prefetch-factor 2 --include-background-negatives --pin-memory --persistent-workers --patch-size 64 --device cuda
```

不使用负样本时删除 `--include-background-negatives`。CPU 时可使用 `--device cpu --num-workers 0 --no-pin-memory --no-persistent-workers`。

## 主要参数

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--input` | 必填 | Top-K 图 `.npz` |
| `--cool-input` | 必填 | `.cool` 文件 |
| `--labels` | `data/datasets` | Excel 标签目录 |
| `--output-dir` | `data/region_embeddings` | 输出目录 |
| `--epochs` | 30 | 训练轮数 |
| `--batch-size` | 32 | batch 大小 |
| `--hidden-dim` | 32 | GNN 隐藏维度 |
| `--embedding-dim` | 16 | embedding 维度，须为 4 的倍数 |
| `--classifier-hidden-dim` | 32 | head 隐藏维度 |
| `--patch-size` | 32 | CNN patch 边长 |
| `--lr` | 0.001 | 学习率 |
| `--test-fraction` | 0.15 | 测试比例 |
| `--num-workers` | 4 | DataLoader worker 数 |
| `--prefetch-factor` | 2 | worker 预取 batch 数 |
| `--opcid-threshold` | 0.55 | OPCID 阈值 |
| `--chin-threshold` | 0.50 | CHIN 阈值 |
| `--chid-threshold` | 0.40 | CHID 阈值 |
| `--scan` | 关闭 | 训练后扫描全图 |
| `--scan-window` | 0 | 扫描窗口，0 使用正样本长度中位数 |
| `--scan-step` | 500 | 扫描步长 |
| `--novelty-threshold` | 0.55 | 新候选阈值 |

## 输出

`<stem>_region_model.pt` 保存 `specialized_model` 和 `args`，包含 GNN、OPCID CNN、CHIN/CHID CNN、cross-attention 和三个 head 的权重。旧版单一 classifier checkpoint 不能直接加载。

`<stem>_region_test.npz` 包含 `embedding`、`labels`、`start`、`end` 和 JSON 格式 `metadata`。

加入 `--scan` 后生成 `<stem>_whole_genome_scan.npy`，字段为 `start`、`end`、`p_opcid`、`p_chin`、`p_chid`、`novelty_score`、`novel_candidate`。

## 指标

数组顺序为 `[OPCID, CHIN, CHID]`。每类指标为：

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
```

`binary_accuracy` 是三个标签位逐元素准确率，不是区域级完全正确率。

