"""
给组长 OPCID 分支做 Grad-CAM：看模型判"这是 OPCID"时，到底盯着接触矩阵的哪一块。

作用：任务一的可解释性要求。证明模型关注的是沿主对角线的方形富集结构，而不是
背景密度或别的噪声。只复用组长脚本的类与函数（import），不改组长任何代码。

用法（仓库根目录）：
python model/opcid/05_gradcam_region.py \
  --checkpoint data/region_embeddings/GSE272159_37C_rep1.mapq_30.10_top40_region_model.pt \
  --graph data/graphs/GSE272159_37C_rep1.mapq_30.10_top40.npz \
  --cool data/datasets/GSE272159_37C_rep1.mapq_30.10.cool \
  --n 6
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(r"D:\projects\deep_learning_tju")
sys.path.insert(0, str(REPO / "model"))

from region_GNN_optimized_none_negative import (  # noqa: E402 复用组长脚本，不改它
    SpecializedRegionModel,
    RegionDataset,
    build_positive_regions,
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def gradcam_opcid(model, sample, device, patch_size):
    """对一个区域的 OPCID 分支算 Grad-CAM，返回 (cam_list, patches)。"""
    x = sample["x"].to(device)
    edge_index = sample["edge_index"].to(device)
    edge_weight = sample["edge_weight"].to(device)
    patches = [p.to(device) for p in sample["patches"]]

    model.eval()
    acts = {}

    def make_hook(i):
        def h(_m, _inp, out):
            out.retain_grad()           # 记住这一层的梯度
            acts[i] = out
        return h

    # 组长 OPCID 分支：三个尺度各走一个 ContactCNN，embedding 取平均再进 opcid_head。
    # 我们 hook 每个 ContactCNN 最后一个卷积层 features[8]（64 通道，还保留空间位置）。
    hooks = [
        model.opcid_cnn[i].features[8].register_forward_hook(make_hook(i))
        for i in range(len(patches))
    ]
    opcid = torch.stack(
        [br(p[0:1]).squeeze(0) for br, p in zip(model.opcid_cnn, patches)]
    ).mean(dim=0)
    logit = model.opcid_head(opcid)     # 0 维标量，可直接 backward
    logit.backward()

    cams = []
    for i in range(len(patches)):
        A = acts[i]                      # (1, 64, h, w)
        grad = A.grad                    # (1, 64, h, w)
        alpha = grad.mean(dim=(0, 2, 3), keepdim=True)          # 每个通道的全局平均梯度
        cam = torch.relu((alpha * A).sum(dim=1))[0]              # (h, w)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0), size=(patch_size, patch_size),
            mode="bilinear", align_corners=False,
        )[0, 0].detach().cpu().numpy()
        cams.append(cam)

    for h in hooks:
        h.remove()
    return cams, patches


def plot_region(cams, patches, start, end, out_path, scales=(250, 500, 1000)):
    fig, axes = plt.subplots(1, len(scales), figsize=(4.2 * len(scales), 4.2))
    for i, (ax, cam, p, s) in enumerate(zip(axes, cams, patches, scales)):
        bg = p[0, 0].detach().cpu().numpy()      # 通道0：log1p 原始接触强度
        ax.imshow(bg, cmap="Greys", origin="lower")
        ax.imshow(cam, cmap="jet", alpha=0.55, origin="lower")
        ax.set_title(f"尺度 {s} bp", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"OPCID Grad-CAM  start={start}  end={end}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="组长用 rep1 训好的 _region_model.pt")
    ap.add_argument("--graph", required=True, help="Top-K 图 npz")
    ap.add_argument("--cool", required=True, help=".cool")
    ap.add_argument("--labels", default=str(REPO / "data/datasets"))
    ap.add_argument("--patch-size", type=int, default=32)
    ap.add_argument("--n", type=int, default=6, help="取多少个真 OPCID 区域")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", default=str(REPO / "model/opcid/data_out/gradcam_region"))
    args = ap.parse_args()

    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cargs = ckpt["args"]

    with np.load(args.graph, allow_pickle=False) as ar:
        arrays = {k: np.ascontiguousarray(ar[k]) for k in ("x", "node_start", "node_end", "edge_index", "edge_weight")}
    array_tensors = {k: torch.from_numpy(v) for k, v in arrays.items()}
    genome_end = int(arrays["node_end"].max())

    regions = build_positive_regions(Path(args.labels), genome_end)
    opcid_idx = [i for i, (_s, _e, t) in enumerate(regions) if t[0] == 1]
    print(f"OPCID 区域总数: {len(opcid_idx)}，本次取前 {min(args.n, len(opcid_idx))} 个")

    model = SpecializedRegionModel(
        arrays["x"].shape[1], cargs["hidden_dim"], cargs["embedding_dim"], cargs["classifier_hidden_dim"]
    ).to(device)
    model.load_state_dict(ckpt["specialized_model"])
    model.eval()

    ds = RegionDataset(regions, array_tensors, Path(args.cool), args.patch_size)

    made = 0
    for idx in opcid_idx[: args.n]:
        sample = ds[idx]
        if sample is None:
            continue
        start, end = sample["start"], sample["end"]
        try:
            cams, patches = gradcam_opcid(model, sample, device, args.patch_size)
        except RuntimeError as e:
            print(f"skip {start}-{end}: {e}")
            continue
        out_path = outdir / f"gradcam_OPCID_{start}_{end}.png"
        plot_region(cams, patches, start, end, out_path)
        made += 1
        print(f"已保存: {out_path}")
    print(f"共生成 {made} 张 Grad-CAM 图")


if __name__ == "__main__":
    main()
