"""
跨重复验证：用 rep1 训练的模型在 rep2 上评估，并扫描 OPCID 阈值。

作用：加载组长用 rep1 训练好的多分支模型权重，在第二个独立生物学重复
rep2 的 .cool + Top-K 图上做推理，评估 OPCID/CHIN/CHID 在 rep2 上的表现。
这符合作业指导书"真实结构应在两个独立生物学重复中均能观察到"的要求。

只复用组长脚本的类与函数（import），不修改组长任何代码。

用法示例（在仓库根目录运行）：
python model/opcid/cross_rep_validate.py \
  --checkpoint data/region_embeddings/GSE272159_37C_rep1.mapq_30.10_top40_region_model.pt \
  --graph <rep2的top40图.npz> \
  --cool <rep2的.cool>
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"D:\projects\deep_learning_tju")
sys.path.insert(0, str(REPO / "model"))

from region_GNN_optimized_none_negative import (  # noqa: E402 复用组长脚本，不改它
    SpecializedRegionModel,
    RegionDataset,
    build_positive_regions,
    evaluate_with_thresholds,
    _make_loader,
    _move_sample_to_device,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="组长用 rep1 训好的 _region_model.pt")
    ap.add_argument("--graph", required=True, help="rep2 的 Top-K 图 npz")
    ap.add_argument("--cool", required=True, help="rep2 的 .cool")
    ap.add_argument("--labels", default=str(REPO / "data/datasets"))
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    # OPCID 阈值扫描：不给 --opcid-threshold 时自动扫描一组阈值。
    ap.add_argument("--opcid-threshold", type=float, default=None,
                    help="只测单个 OPCID 阈值；不传则自动扫描 0.30~0.60")
    ap.add_argument("--chin-threshold", type=float, default=0.5)
    ap.add_argument("--chid-threshold", type=float, default=0.4)
    args = ap.parse_args()

    device = torch.device(args.device)

    # ---- 1. 加载组长用 rep1 训练好的模型权重 ----
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cargs = ckpt["args"]
    print(json.dumps({"loaded_checkpoint": str(args.checkpoint), "train_args_sample_mode": cargs.get("sample_mode")}, ensure_ascii=False))

    # ---- 2. 加载 rep2 的图（与组长脚本加载方式一致） ----
    with np.load(args.graph, allow_pickle=False) as ar:
        arrays = {k: np.ascontiguousarray(ar[k]) for k in ("x", "node_start", "node_end", "edge_index", "edge_weight")}
    # 传给 RegionDataset 的 shared_arrays 需要 torch tensor（其内部会 .numpy()）
    array_tensors = {k: torch.from_numpy(v) for k, v in arrays.items()}
    genome_end = int(arrays["node_end"].max())

    # ---- 3. 从 Excel 读取标注区域（与组长脚本一致） ----
    regions = build_positive_regions(Path(args.labels), genome_end)

    # ---- 4. 构建模型并加载 rep1 权重 ----
    model = SpecializedRegionModel(
        arrays["x"].shape[1], cargs["hidden_dim"], cargs["embedding_dim"], cargs["classifier_hidden_dim"]
    ).to(device)
    model.load_state_dict(ckpt["specialized_model"])
    model.eval()

    # ---- 5. 用 rep2 数据构建 dataset 并推理 ----
    ds = RegionDataset(regions, array_tensors, Path(args.cool), args.patch_size)
    loader = _make_loader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=False, persistent_workers=False, prefetch_factor=2,
    )

    logits_list, label_list = [], []
    with torch.no_grad():
        for batch in loader:
            for sample in batch:
                graph, patches, label = _move_sample_to_device(sample, device, False)
                lg = model(graph, patches)          # [OPCID, CHIN, CHID] 单样本
                logits_list.append(lg.cpu())
                label_list.append(label.cpu())
    # ---- 6. OPCID 阈值扫描（CHIN/CHID 阈值保持组长默认） ----
    # evaluate_with_thresholds 内部会 torch.stack，因此直接传逐样本列表。
    opcid_ts = [args.opcid_threshold] if args.opcid_threshold is not None \
        else [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    rows = []
    for ot in opcid_ts:
        thresholds = (ot, args.chin_threshold, args.chid_threshold)
        stats = evaluate_with_thresholds(logits_list, label_list, thresholds)
        p, r, f1 = stats["precision"], stats["recall"], stats["f1"]
        rows.append({
            "opcid_threshold": ot,
            "opcid_precision": float(p[0]), "opcid_recall": float(r[0]), "opcid_f1": float(f1[0]),
            "chin_f1": float(f1[1]), "chid_f1": float(f1[2]),
        })
        print(json.dumps({
            "opcid_threshold": ot,
            "opcid": [round(p[0], 3), round(r[0], 3), round(f1[0], 3)],   # [precision, recall, f1]
            "chin_f1": round(f1[1], 3), "chid_f1": round(f1[2], 3),
        }, ensure_ascii=False))

    best = max(rows, key=lambda x: x["opcid_f1"])
    print(json.dumps({
        "scan_best_opcid_threshold": best["opcid_threshold"],
        "scan_best_opcid": [round(best["opcid_precision"], 3), round(best["opcid_recall"], 3), round(best["opcid_f1"], 3)],
        "rep2_regions": int(len(label_list)),
        "sample_mode": "cross_replicate_rep1_train_rep2_test",
    }, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
