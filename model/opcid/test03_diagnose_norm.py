"""对比几种归一化下正负样本的可分性（AUC）。"""
import os, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import cooler
import scipy.sparse as sp

BASE = r"C:\Users\ZhuanZ1\Doubao\chats\2026-09-28\new-chat\opcid_task"
COOL = BASE + r"\datasets\GSE272159_37C_rep1.mapq_30.10.cool"
NPZ  = BASE + r"\data_out\rep1_data.npz"

def build_binned_matrix(cool_path, n_bins=46416):
    c = cooler.Cooler(cool_path)
    M = c.matrix(balance=False, sparse=True)[:].tocoo()
    keep = (M.row < n_bins * 10) & (M.col < n_bins * 10)
    d, r, col = M.data[keep], M.row[keep], M.col[keep]
    A = sp.coo_matrix((d, (r // 10, col // 10)), shape=(n_bins, n_bins), dtype=np.float64).tocsr()
    return A

def cut(A, cb, W=200):
    c0, c1 = cb - W // 2, cb + W // 2
    n = A.shape[0]
    win = np.zeros((W, W))
    i0, i1 = max(c0, 0), min(c1, n)
    sub = A[i0:i1, i0:i1].toarray()
    win[i0 - c0: i0 - c0 + sub.shape[0], i0 - c0: i0 - c0 + sub.shape[1]] = sub
    return win

def oe(w):
    out = w.copy(); n = out.shape[0]
    for dd in range(1, n):
        m = float(np.mean(out.diagonal(dd)))
        if m > 0:
            r = np.arange(n - dd); c_ = r + dd
            out[r, c_] = out[r, c_] / m; out[c_, r] = out[c_, r] / m
    return out

def minmax(x):
    a, b = x.min(), x.max()
    return (x - a) / (b - a + 1e-8)

d = np.load(NPZ, allow_pickle=False)
centers = [int(c) // 100 for c in d["coords"][:, 2]]
labels = d["labels"]
print("读并聚合 .cool ...")
A = build_binned_matrix(COOL)

schemes = {}
for cb in centers:
    w = cut(A, cb)
    schemes.setdefault("raw", []).append(w)
    schemes.setdefault("oe", []).append(oe(w))
schemes["raw"] = np.stack(schemes["raw"]); schemes["oe"] = np.stack(schemes["oe"])
schemes["log1p"] = np.stack([minmax(np.log1p(w)) for w in schemes["raw"]])
schemes["oe_log1p"] = np.stack([minmax(np.log1p(w)) for w in schemes["oe"]])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.decomposition import PCA
cv = StratifiedKFold(5, shuffle=True, random_state=0)

def stat_feats(imgs):
    n = imgs.shape[1]; out = []
    for i in range(len(imgs)):
        w = imgs[i]; u = np.triu(w, 1)
        out.append([w.mean(), w.max(), np.percentile(w, 99.5), np.std(w),
                    np.trace(w) / n, float(u[u > 0].mean()) if (u > 0).any() else 0])
    return np.array(out)

for name, imgs in schemes.items():
    f = stat_feats(imgs)
    a1 = cross_val_score(LogisticRegression(max_iter=2000), f, labels, cv=cv, scoring="roc_auc").mean()
    X = imgs.reshape(len(imgs), -1)
    P = PCA(n_components=20, random_state=0).fit_transform(X)
    a2 = cross_val_score(LogisticRegression(max_iter=2000), P, labels, cv=cv, scoring="roc_auc").mean()
    print(f"归一化[{name:<9}]  统计特征AUC={a1:.3f}   PCA+LR AUC={a2:.3f}")
