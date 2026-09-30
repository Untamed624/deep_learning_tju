"""
构建 OPCID 二分类数据集：按 100bp 聚合切窗。

做什么：
  1. 用 cooler 读取 .cool 的 10bp 接触矩阵；
  2. 把每 10 个 10bp bin 求和聚合成 100bp bin（共 46416 个）；
  3. 以每个 OPCID 坐标为中心，切出一个 WINDOW_BINS×WINDOW_BINS 的局部窗口（默认 200×200 = 20kb）；
     正样本 = 68 个 OPCID 窗口；
  4. 在基因组上随机采 68 个位置作为负样本（1:1），默认避开 OPCID 防止污染标签；
  5. 每个窗口做 O/E 归一化（按对角距离带，各带均值=1），弱信号 OPCID 的矩形会显出来；
  6. 同一个窗口坐标分别从 rep1 / rep2 各切一份，存成两个 .npz：
       data_out/rep1_data.npz  -> 用于训练/留出验证
       data_out/rep2_data.npz  -> 用于跨生物学重复验证（作业加分项）

运行：python 01_build_dataset.py
输出：data_out/rep1_data.npz, data_out/rep2_data.npz（可被 02 训练脚本直接读）

依赖：cooler, numpy, openpyxl, scipy（Anaconda 里都已就绪）
"""
from __future__ import annotations
import os, json
import numpy as np
import cooler
import scipy.sparse as sp

# ========================= 配置区（改这里）=========================
BASE       = r"C:\Users\ZhuanZ1\Doubao\chats\2026-09-28\new-chat\opcid_task"
COOL_REP1  = BASE + r"\datasets\GSE272159_37C_rep1.mapq_30.10.cool"
COOL_REP2  = BASE + r"\datasets\GSE272159_37C_rep2.mapq_30.10.cool"
OPCID_XLSX = BASE + r"\datasets\OPCID_data.xlsx"
CHIN_XLSX  = BASE + r"\datasets\CHIN_data.xlsx"
CHID_XLSX  = BASE + r"\datasets\CHID_data.xlsx"
OUT_DIR    = BASE + r"\data_out"

BIN_SIZE     = 100      # 聚合后的分辨率（bp）
WINDOW_BINS  = 200      # 窗口边长（单位：100bp bin）= 20kb
HALF         = WINDOW_BINS // 2
N_POS        = None     # 正样本数量，None 表示取全部 68 个 OPCID
N_NEG        = None     # 负样本数量，None 表示与正样本 1:1
SEED         = 42       # 随机种子（保证可复现）
AVOID_OPCID    = True   # 负样本窗口避开 OPCID（防污染标签）
AVOID_CHIN_CHID = False # 后续改良：设为 True 则负样本也避开 CHIN/CHID
USE_LOG1P     = True    # O/E 后是否再取 log1p(1+x) 增强弱信号对比
# ================================================================

def load_structs(path, idcol, startcol, endcol, centercol=None):
    """读 xlsx 结构坐标表，返回 [(id, start_bp, end_bp, center_bp), ...]"""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    idx = {h: i for i, h in enumerate(rows[0])}
    out = []
    for r in rows[1:]:
        if r[idx[startcol]] is None:
            continue
        start = int(r[idx[startcol]]); end = int(r[idx[endcol]])
        center = int(r[idx[centercol]]) if centercol else (start + end) // 2
        out.append((str(r[idx[idcol]]), start, end, center))
    return out


def bin100_of(bp: int) -> int:
    """bp 坐标 -> 100bp bin 索引"""
    return bp // BIN_SIZE


def intervals_overlap(win_start_bp, win_end_bp, known):
    """窗口区间是否与任一已知区间重叠（半开区间）"""
    for (s, e) in known:
        if win_start_bp < e and s < win_end_bp:
            return True
    return False


def build_binned_matrix(cool_path: str, n_bins: int):
    """读 .cool，聚合 10bp -> 100bp，返回 (46416, 46416) 稀疏接触矩阵。"""
    c = cooler.Cooler(cool_path)
    M = c.matrix(balance=False, sparse=True)[:].tocoo()
    keep = (M.row < n_bins * 10) & (M.col < n_bins * 10)  # 丢弃末尾不足10个的10bp bin（row/col是10bp索引）
    data, row, col = M.data[keep], M.row[keep], M.col[keep]
    R = row // 10                                       # 10 个 10bp bin 合并为 1 个 100bp bin
    C = col // 10
    A = sp.coo_matrix((data, (R, C)), shape=(n_bins, n_bins), dtype=np.float64).tocsr()
    return A


def cut_window(A, center_bin: int):
    """以 center_bin 为中心切 WINDOW_BINS×WINDOW_BINS 窗口；越界补 0。返回 dense float64"""
    c0, c1 = center_bin - HALF, center_bin + HALF
    n = A.shape[0]
    win = np.zeros((WINDOW_BINS, WINDOW_BINS), dtype=np.float64)
    i0, i1 = max(c0, 0), min(c1, n)                     # 有效行/列
    sub = A[i0:i1, i0:i1].toarray() if i1 > i0 else np.zeros((0, 0))
    win[i0 - c0 : i0 - c0 + sub.shape[0], i0 - c0 : i0 - c0 + sub.shape[1]] = sub
    return win


def oe_normalize(win: np.ndarray) -> np.ndarray:
    """O/E：按对角距离带归一，使每带的背景均值≈1，结构富集>1。对角线(d=0)不参与。"""
    out = win.copy()
    n = out.shape[0]
    for d in range(1, n):
        m = float(np.mean(out.diagonal(d)))      # 距离 d 的带均值
        if m > 0:
            rows = np.arange(n - d)
            cols = rows + d
            out[rows, cols] = out[rows, cols] / m     # 上三角
            out[cols, rows] = out[cols, rows] / m     # 下三角（保持对称）
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    opcid = load_structs(OPCID_XLSX, "OPCID_ID", "Start", "End")
    known_opcid = [(s, e) for _, s, e, _ in opcid]
    known_chin  = [(s, e) for _, s, e, _ in load_structs(CHIN_XLSX, "CHIN_ID", "Start", "End")]
    known_chid  = [(s, e) for _, s, e, _ in load_structs(CHID_XLSX, "CHID_ID", "Start", "End")]
    known_all   = known_opcid + known_chin + known_chid

    n_pos = N_POS if N_POS else len(opcid)
    n_neg = N_NEG if N_NEG else n_pos                   # 默认 1:1
    rng = np.random.default_rng(SEED)

    # ---------- 确定所有样本的窗口中心（bp 与 bin） ----------
    # 正样本中心
    pos = opcid[:n_pos]
    centers = []                                        # (center_bp, center_bin, is_opcid)
    for _, s, e, c in pos:
        centers.append((c, bin100_of(c), 1))
    # 负样本中心（纯随机 + 可选避让）
    while sum(1 for _, _, l in centers if l == 0) < n_neg:
        cb = int(rng.integers(HALF, 46416 - HALF))
        cbp = cb * BIN_SIZE
        ws, we = cbp - HALF * BIN_SIZE, cbp + HALF * BIN_SIZE
        if AVOID_OPCID and intervals_overlap(ws, we, known_opcid):
            continue
        if AVOID_CHIN_CHID and intervals_overlap(ws, we, known_all):
            continue
        centers.append((cbp, cb, 0))
    neg_centers = [c for c in centers if c[2] == 0][:n_neg]
    centers = [c for c in centers if c[2] == 1] + neg_centers
    assert len(centers) == n_pos + n_neg

    labels = np.array([l for _, _, l in centers], dtype=np.int64)
    coords = np.array([[s, s, c] for c, b, _ in centers], dtype=np.int64)  # 占位，下面填真实起止
    # 填真实窗口起止（bp）
    coords[:, 0] = [c - HALF * BIN_SIZE for c, _, _ in centers]
    coords[:, 1] = [c + HALF * BIN_SIZE for c, _, _ in centers]

    meta = {
        "schema": 1, "bin_size_bp": BIN_SIZE, "window_bins": WINDOW_BINS,
        "n_positive": n_pos, "n_negative": n_neg, "seed": SEED,
        "oe_norm": True, "log1p": USE_LOG1P,
        "negative_policy": ("avoid_opcid" if AVOID_OPCID else "pure_random")
                           + ("" if not AVOID_CHIN_CHID else "_avoid_chin_chid"),
        "center_bins": [int(b) for _, b, _ in centers],
    }

    # ---------- 从 rep1 / rep2 各切一份 ----------
    for rep_name, cool_path, out_name in [
        ("rep1", COOL_REP1, "rep1_data.npz"),
        ("rep2", COOL_REP2, "rep2_data.npz"),
    ]:
        print(f"[{rep_name}] 读取并聚合 .cool ...", flush=True)
        A = build_binned_matrix(cool_path, 46416)
        images = np.empty((len(centers), WINDOW_BINS, WINDOW_BINS), dtype=np.float32)
        for i, (_, cb, _) in enumerate(centers):
            w = oe_normalize(cut_window(A, cb))
            if USE_LOG1P:
                w = np.log1p(w)
            images[i] = w
        out_path = os.path.join(OUT_DIR, out_name)
        np.savez_compressed(out_path, images=images, labels=labels, coords=coords,
                            metadata=np.array(json.dumps(meta, ensure_ascii=False)))
        print(f"[{rep_name}] 完成 -> {out_path}  形状={images.shape}  正={int((labels==1).sum())} 负={int((labels==0).sum())}", flush=True)


if __name__ == "__main__":
    main()
