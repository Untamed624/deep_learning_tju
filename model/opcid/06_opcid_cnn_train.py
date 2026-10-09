"""
独立的 OPCID CNN 训练脚本。

在"组长 OPCID 分支同款架构 + 同款数据构建"下单独训练 OPCID 二分类，
用于和组长多标签模型里的 OPCID 分支效果做对比。数据管线复用组长脚本
（read_interval_labels / multiscale_patches / _contact_channels / ContactCNN），
保证同一套数据、同一套 patch、同一套 CNN，只有"任务设定"不同。

模型：3 个 ContactCNN（对应 250/500/1000 bp 三尺度方形 patch）→ embedding 平均
     → 二分类 head → sigmoid。对应组长 SpecializedRegionModel 的 opcid 分支，
     但去掉 GNN，纯 CNN。

负样本策略（--negative-strategy）：
  random_bg : 随机背景 1:1（简单任务，对应早期思路）
  annotated : 用 CHIN/CHID 真实结构作负样本 1:1（难任务，与组长多标签同难度）

运行示例：
  python model/opcid/06_opcid_cnn_train.py \
    --input data/graphs/GSE272159_37C_rep1.mapq_30.10_top40.npz \
    --cool-input data/datasets/GSE272159_37C_rep1.mapq_30.10.cool \
    --labels data/datasets \
    --output-dir data/region_embeddings \
    --negative-strategy random_bg \
    --device cpu
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP 冲突绕过
import sys
from pathlib import Path
import argparse
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 让脚本能 import 到 model/ 下的组长模块
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from region_GNN import ContactCNN, multiscale_patches
from GNN import read_interval_labels


# ============================== 数据 ==============================

def _load_node_arrays(npz_path):
    """从图 npz 读切窗口需要的 node_start/node_end。"""
    with np.load(npz_path, allow_pickle=False) as archive:
        return {
            "node_start": archive["node_start"].astype(np.int64),
            "node_end": archive["node_end"].astype(np.int64),
        }


def build_samples(labels_dir, genome_end, seed, strategy, max_negatives=-1):
    """构建 OPCID 二分类样本 [(start, end, label)]，label ∈ {0,1}。

    random_bg : 负样本=随机背景（避开全部标注），数量=OPCID 数（1:1）
    annotated : 负样本=CHIN/CHID 真实结构（避开与 OPCID 重叠）
                max_negatives=-1 时取全部（1:4 不平衡，与组长同口径）；
                >0 时只取前 N 个。
    """
    intervals = read_interval_labels(labels_dir)
    opcid = [(s, e) for _, s, e, l in intervals if l == "OPCID"]
    chin = [(s, e) for _, s, e, l in intervals if l == "CHIN"]
    chid = [(s, e) for _, s, e, l in intervals if l == "CHID"]

    positives = [(s, e, 1.0) for s, e in opcid]

    if strategy == "random_bg":
        occupied = opcid + chin + chid
        rng = np.random.default_rng(seed)
        negatives, tries = [], 0
        while len(negatives) < len(positives) and tries < max(1, len(positives) * 2000):
            tries += 1
            s, e, _ = positives[int(rng.integers(len(positives)))]
            length = e - s
            if genome_end <= length:
                break
            start = int(rng.integers(0, genome_end - length + 1))
            end = start + length
            if any(start < oe and end > os for os, oe in occupied):
                continue
            negatives.append((start, end, 0.0))
            occupied.append((start, end))
        samples = positives + negatives
    elif strategy == "annotated":
        neg = [
            (s, e) for s, e in (chin + chid)
            if not any(s < oe and e > os for os, oe in opcid)
        ]
        if max_negatives > 0:
            neg = neg[:max_negatives]
        samples = positives + [(s, e, 0.0) for s, e in neg]
    else:
        raise ValueError(f"unknown --negative-strategy: {strategy}")

    if not samples:
        raise ValueError("no samples built; check labels dir")
    return samples


class OpcidDataset(Dataset):
    """每样本三尺度方形 patch，label 为 OPCID 0/1 标量。

    patch 在初始化时一次性切好缓存，训练循环不再重复切，避免 CPU 上每
    epoch 重切导致极慢。
    """

    def __init__(self, samples, cool_path, arrays, patch_size):
        self.cool_path = str(cool_path)
        self.arrays = arrays
        self.patch_size = patch_size
        self._precompute(samples)

    def _precompute(self, samples):
        import cooler
        c = cooler.Cooler(self.cool_path)
        matrix = c.matrix(balance=False, sparse=False)
        self.samples = []
        for start, end, label in samples:
            patches = multiscale_patches(
                matrix, self.arrays, (start, end), (250, 500, 1000),
                self.patch_size, torch.device("cpu"),
            )
            if patches is None:
                continue
            self.samples.append({
                "patches": tuple(patches),
                "label": torch.tensor(float(label)),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _collate(samples):
    return [s for s in samples if s is not None]


# ============================== 模型 ==============================

class OpcidCNN(torch.nn.Module):
    """组长 OPCID 分支同款：三尺度 ContactCNN → 平均 → 二分类 head（纯 CNN）。"""

    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.opcid_cnn = torch.nn.ModuleList(
            [ContactCNN(embedding_dim) for _ in (250, 500, 1000)]
        )
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim),
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, patches):
        embs = [br(p[0:1]).squeeze(0) for br, p in zip(self.opcid_cnn, patches)]
        emb = torch.stack(embs).mean(dim=0)
        return self.head(emb).squeeze(-1)


# ============================== 评估 ==============================

def evaluate(model, loader, device, threshold):
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            for sample in batch:
                if sample is None:
                    continue
                patches = [p.to(device) for p in sample["patches"]]
                logits.append(model(patches).detach().cpu())
                labels.append(sample["label"].detach().cpu())
    if not logits:
        return None
    logits = torch.stack(logits)
    labels = torch.stack(labels)
    prob = torch.sigmoid(logits)
    pred = (prob >= threshold).float()
    tp = (pred * labels).sum().item()
    fp = (pred * (1 - labels)).sum().item()
    fn = ((1 - pred) * labels).sum().item()
    tn = ((1 - pred) * (1 - labels)).sum().item()
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    acc = (pred == labels).float().mean().item()
    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def scan_thresholds(model, loader, device):
    """在 0.1~0.9 扫阈值，返回最优 F1 和最优 recall 对应的评估。"""
    best_f1, best_recall = None, None
    for t in [round(0.10 + 0.05 * i, 2) for i in range(17)]:
        r = evaluate(model, loader, device, t)
        if r is None:
            continue
        if best_f1 is None or r["f1"] > best_f1["f1"]:
            best_f1 = r
        if best_recall is None or r["recall"] > best_recall["recall"]:
            best_recall = r
    return best_f1, best_recall


# ============================== 主流程 ==============================

def main():
    root = Path(__file__).resolve().parents[2]      # 仓库根目录
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="top-k 图 .npz（提供 node_start/node_end）")
    parser.add_argument("--cool-input", type=Path, required=True, help="原始 .cool，用于切接触 patch")
    parser.add_argument("--labels", type=Path, default=root / "data/datasets")
    parser.add_argument("--output-dir", type=Path, default=root / "data/region_embeddings")
    parser.add_argument("--negative-strategy", choices=["random_bg", "annotated"], default="random_bg")
    parser.add_argument("--max-negatives", type=int, default=-1, help="annotated 负样本数量上限，-1=全量（1:4 不平衡）")
    parser.add_argument("--test-input", type=Path, default=None, help="rep2 图 npz（node_start/node_end），跨重复测试用")
    parser.add_argument("--test-cool-input", type=Path, default=None, help="rep2 .cool；提供后进入跨重复：rep1 全量训练、rep2 全量测试")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5, help="固定评估阈值")
    parser.add_argument("--scan", action="store_true", help="在 0.1~0.9 扫阈值并报告最优")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 读图（切窗口只需要 node_start/node_end）----
    arrays = _load_node_arrays(args.input)
    genome_end = int(arrays["node_end"].max())
    test_arrays = _load_node_arrays(args.test_input) if args.test_input else arrays

    # ---- 构建样本并划分 ----
    samples = build_samples(args.labels, genome_end, args.seed, args.negative_strategy, args.max_negatives)
    cross_rep = args.test_cool_input is not None
    if cross_rep:
        # 跨重复：rep1 全量训练、rep2 全量测试（同一批标注区间，换矩阵切 patch）
        train_samples = samples
        test_samples = samples
        test_cool = args.test_cool_input
    else:
        order = np.random.default_rng(args.seed).permutation(len(samples))
        split = max(1, int(round(len(samples) * (1 - args.test_fraction))))
        train_samples = [samples[i] for i in order[:split]]
        test_samples = [samples[i] for i in order[split:]]
        test_cool = args.cool_input
    if not train_samples or not test_samples:
        raise ValueError("train/test split produced an empty partition")

    train_ds = OpcidDataset(train_samples, args.cool_input, arrays, args.patch_size)
    test_ds = OpcidDataset(test_samples, test_cool, test_arrays, args.patch_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate)

    # ---- 模型 ----
    model = OpcidCNN(args.embedding_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = torch.nn.BCEWithLogitsLoss()

    print(json.dumps({
        "strategy": args.negative_strategy,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "n_positive": sum(1 for _, _, l in samples if l == 1.0),
        "n_negative": sum(1 for _, _, l in samples if l == 0.0),
        "device": str(device),
    }, ensure_ascii=False))

    # ---- 训练 ----
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in train_loader:
            if not batch:
                continue
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for sample in batch:
                patches = [p.to(device) for p in sample["patches"]]
                label = sample["label"].to(device)
                losses.append(criterion(model(patches), label))
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1
        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"epoch {epoch}/{args.epochs}  loss={total_loss/max(steps,1):.4f}", flush=True)

    # ---- 评估 ----
    report = {"fixed": evaluate(model, test_loader, device, args.threshold)}
    if args.scan:
        best_f1, best_recall = scan_thresholds(model, test_loader, device)
        report["best_f1"] = best_f1
        report["best_recall"] = best_recall

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.input).stem
    torch.save({
        "opcid_cnn_state": model.state_dict(),
        "args": vars(args),
        "report": report,
    }, args.output_dir / f"{stem}_opcid_cnn_independent.pt")

    print(json.dumps({"report": report}, ensure_ascii=False, indent=2))
    print(f"saved -> {args.output_dir / (stem + '_opcid_cnn_independent.pt')}")


if __name__ == "__main__":
    main()
