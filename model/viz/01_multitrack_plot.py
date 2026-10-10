"""多轨道接触频率与结构分布可视化（第二版）。

按任务三要求绘制基因组浏览器风格的四面板图：
顶部：各实验条件接触强度曲线（平滑后）+ 两条阈值虚线
中间：多条件平均信号强度（灰色局部填充）
基因轨道：蓝色方块标出基因位置，标注基因名
结构轨道：橙色方块标出 CHIN / OPCID 区间，标注编号

所有面板共享同一 X 轴（基因组位置 bp）。
"""

import argparse
import os

import cooler
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from matplotlib.gridspec import GridSpec

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def smooth_curve(x, window=50):
    """移动平均平滑曲线，window 为 bin 数（1 bin = 10bp）。"""
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def load_bin_contact_strength(cool_path, chrom="NC_000913.3"):
    """读取 .cool 文件，计算每个 bin 的归一化接触强度（sqrt(row_sum / total_reads)）。

    归一化消除不同条件测序深度差异。
    返回:
        positions: 每个 bin 的基因组起始位置 (bp)
        strength: 归一化后的 sqrt 接触强度
    """
    c = cooler.Cooler(str(cool_path))
    mat = c.matrix(balance=False, sparse=True).fetch(chrom)
    row_sum = np.array(mat.sum(axis=1)).flatten()
    total_reads = row_sum.sum()  # 总 reads 数，用于归一化
    # 归一化后开平方根
    strength = np.sqrt(row_sum / total_reads * 1e6)  # 乘以 1e6 方便读数
    bins = c.bins().fetch(chrom)
    positions = bins["start"].values
    return positions, strength


def load_genes(gff_path, chrom="NC_000913.3"):
    """读取 GFF 文件，提取基因区间及名称。"""
    genes = []
    with open(gff_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue
            seqname, source, feature, start, end, score, strand, frame, attrs = parts
            if seqname != chrom or feature != "gene":
                continue
            gene_name = ""
            for attr in attrs.split(";"):
                if attr.startswith("Name="):
                    gene_name = attr.split("=")[1]
                    break
            genes.append((int(start), int(end), gene_name))
    return genes


def load_regions_from_xlsx(xlsx_path):
    """读取结构标注 xlsx，提取区间及编号。

    返回:
        list of (start, end, name)
    """
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    regions = []
    header = None
    for row in ws.iter_rows(values_only=True):
        if header is None:
            header = [str(c).lower() if c else "" for c in row]
            continue
        start = end = name = None
        for i, val in enumerate(row):
            col = header[i] if i < len(header) else ""
            if col in ("start", "begin"):
                start = val
            elif col in ("end", "stop"):
                end = val
            elif "id" in col:
                name = str(val) if val else ""
        if start is None or end is None:
            continue
        try:
            regions.append((int(start), int(end), name))
        except (ValueError, TypeError):
            continue
    wb.close()
    return regions


def plot_multitrack(
    window_start, window_end,
    conditions, genes,
    chin_regions, opcid_regions, chid_regions,
    output_path,
):
    """绘制单个窗口的四面板多轨道图。"""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(4, 1, height_ratios=[3, 1.2, 0.6, 0.6], hspace=0.08)

    # ---- 面板 1：各条件接触强度曲线 + 阈值虚线 ----
    ax1 = fig.add_subplot(gs[0])
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    all_smooth = []

    for i, (name, positions, strength) in enumerate(conditions):
        mask = (positions >= window_start) & (positions <= window_end)
        pos = positions[mask]
        sig = strength[mask]
        if len(sig) == 0:
            continue
        # 平滑
        sig_smooth = smooth_curve(sig, window=50)
        all_smooth.append(sig_smooth)
        color = colors[i % len(colors)]
        ax1.plot(pos, sig_smooth, label=name, color=color, linewidth=0.9)

    # 计算阈值虚线（基于所有条件的平均信号）
    if all_smooth:
        min_len = min(len(s) for s in all_smooth)
        avg = np.mean([s[:min_len] for s in all_smooth], axis=0)
        avg_pos = np.linspace(window_start, window_end, min_len)
        # Primary threshold: 均值 + 1倍标准差
        primary_thr = np.mean(avg) + np.std(avg)
        # Secondary threshold: 均值 + 2倍标准差
        secondary_thr = np.mean(avg) + 2 * np.std(avg)
        # 画阈值虚线
        ax1.axhline(y=primary_thr, color="gray", linestyle="--", linewidth=1,
                    label=f"Primary threshold ({primary_thr:.1f})")
        ax1.axhline(y=secondary_thr, color="darkgray", linestyle=":", linewidth=1,
                    label=f"Secondary threshold ({secondary_thr:.1f})")

    ax1.set_ylabel("Sqrt of read number")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_xlim(window_start, window_end)

    # ---- 面板 2：多条件平均信号（灰色局部填充）----
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    if all_smooth:
        min_len = min(len(s) for s in all_smooth)
        avg = np.mean([s[:min_len] for s in all_smooth], axis=0)
        avg_pos = np.linspace(window_start, window_end, min_len)
        ax2.fill_between(avg_pos, avg, color="gray", alpha=0.5)
    ax2.set_ylabel("平均信号")

    # ---- 面板 3：基因轨道（蓝色方块 + 基因名）----
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for g_start, g_end, g_name in genes:
        if g_end < window_start or g_start > window_end:
            continue
        width = g_end - g_start
        ax3.barh(0, width, left=g_start, height=0.6, color="#4472C4", edgecolor="none")
        # 所有基因都标名字，色块内白字；太窄的缩小字体
        if g_name:
            fontsize = 5 if width < 500 else 6 if width < 800 else 7
            ax3.text(g_start + width / 2, 0, g_name, ha="center",
                     va="center", fontsize=fontsize, color="white")
    ax3.set_ylim(-0.8, 0.5)
    ax3.set_yticks([])
    ax3.set_ylabel("基因")

    # ---- 面板 4：结构轨道（CHIN橙色 / CHID紫色 / OPCID黄色 + 编号）----
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    # 标签钳制：把标签位置拉回图内，避免跨图结构的标签跑出边界
    margin = (window_end - window_start) * 0.03  # 边距 3%

    def clamp_label_pos(r_start, r_end):
        """计算标签 x 位置：优先可见区间中心，超出边界就拉到边上。"""
        vis_s = max(r_start, window_start)
        vis_e = min(r_end, window_end)
        center = (vis_s + vis_e) / 2
        # 钳制到图内留边距
        return max(window_start + margin, min(window_end - margin, center))

    # CHIN 橙色（上）
    for r_start, r_end, r_name in chin_regions:
        if r_end < window_start or r_start > window_end:
            continue
        vis_s = max(r_start, window_start)
        vis_e = min(r_end, window_end)
        ax4.barh(0.6, vis_e - vis_s, left=vis_s, height=0.3, color="#ED7D31", edgecolor="none")
        if r_name:
            lx = clamp_label_pos(r_start, r_end)
            ax4.text(lx, 0.8, r_name, ha="center", va="bottom",
                     fontsize=6, color="#ED7D31")
    # CHID 紫色（中）
    for r_start, r_end, r_name in chid_regions:
        if r_end < window_start or r_start > window_end:
            continue
        vis_s = max(r_start, window_start)
        vis_e = min(r_end, window_end)
        ax4.barh(0.15, vis_e - vis_s, left=vis_s, height=0.3, color="#9467BD", edgecolor="none")
        if r_name:
            lx = clamp_label_pos(r_start, r_end)
            ax4.text(lx, 0.35, r_name, ha="center", va="bottom",
                     fontsize=6, color="#9467BD")
    # OPCID 黄色（下）
    for r_start, r_end, r_name in opcid_regions:
        if r_end < window_start or r_start > window_end:
            continue
        vis_s = max(r_start, window_start)
        vis_e = min(r_end, window_end)
        ax4.barh(-0.3, vis_e - vis_s, left=vis_s, height=0.3, color="#FFC000", edgecolor="none")
        if r_name:
            lx = clamp_label_pos(r_start, r_end)
            ax4.text(lx, -0.55, r_name, ha="center", va="top",
                     fontsize=6, color="#B8860B")
    ax4.set_ylim(-0.8, 1.0)
    ax4.set_yticks([-0.3, 0.15, 0.6])
    ax4.set_yticklabels(["OPCID", "CHID", "CHIN"])
    ax4.set_ylabel("结构")

    # X 轴
    ax4.set_xlabel("基因组位置 (bp)")
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    # 标题放在最顶部
    fig.suptitle(f"接触频率与结构分布  {window_start:,} - {window_end:,} bp",
                 y=0.98, fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="多轨道接触频率与结构分布可视化")
    parser.add_argument("--datasets-dir", default="data/datasets")
    parser.add_argument("--output-dir", default="data/region_embeddings/viz")
    parser.add_argument("--window", type=int, default=10000, help="每张图窗口大小 bp")
    parser.add_argument("--regions", type=str, default=None,
                        help="指定区间 start1-end1,start2-end2")
    parser.add_argument("--chrom", default="NC_000913.3")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 条件列表
    conditions_files = [
        ("WT 野生型", "GSE272159_37C_rep1.mapq_30.10.cool"),
        ("博来霉素 bleo", "GSM8950769_bleo_rep1.MG1655.mapq_30.10.cool"),
        ("室温 RT_fix", "GSM8950768_RT_RT_fix_rep1.MG1655.mapq_30.10.cool"),
    ]

    print("加载接触矩阵（归一化 + 稀疏计算）...")
    conditions = []
    for name, fname in conditions_files:
        path = os.path.join(args.datasets_dir, fname)
        if not os.path.exists(path):
            print(f"  跳过: {fname}")
            continue
        print(f"  {name}...")
        positions, strength = load_bin_contact_strength(path, args.chrom)
        conditions.append((name, positions, strength))

    print("加载基因注释...")
    genes = load_genes(os.path.join(args.datasets_dir, "NC_000913.3.gff"), args.chrom)
    print(f"  基因数: {len(genes)}")

    print("加载结构标注...")
    chin_regions = load_regions_from_xlsx(os.path.join(args.datasets_dir, "CHIN_data.xlsx"))
    opcid_regions = load_regions_from_xlsx(os.path.join(args.datasets_dir, "OPCID_data.xlsx"))
    chid_regions = load_regions_from_xlsx(os.path.join(args.datasets_dir, "CHID_data.xlsx"))
    print(f"  CHIN: {len(chin_regions)}, OPCID: {len(opcid_regions)}, CHID: {len(chid_regions)}")

    # 确定区间
    if args.regions:
        regions_to_plot = []
        for r in args.regions.split(","):
            s, e = r.split("-")
            regions_to_plot.append((int(s), int(e)))
    else:
        chrom_length = 4641652
        regions_to_plot = [
            (start, min(start + args.window, chrom_length))
            for start in range(0, chrom_length, args.window)
        ]

    print(f"\n共 {len(regions_to_plot)} 个窗口待绘制...")
    for i, (w_start, w_end) in enumerate(regions_to_plot):
        out_name = f"viss2_{i+1:02d}.png"
        out_path = os.path.join(args.output_dir, out_name)
        plot_multitrack(w_start, w_end, conditions, genes,
                        chin_regions, opcid_regions, chid_regions, out_path)
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(regions_to_plot)}")

    print(f"\n完成！输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
