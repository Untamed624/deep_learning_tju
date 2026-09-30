"""看正负样本特征分布，找为什么难分。"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np

BASE = r"C:\Users\ZhuanZ1\Doubao\chats\2026-09-28\new-chat\opcid_task"
d = np.load(BASE + r"\data_out\rep1_data.npz", allow_pickle=False)
im, lab = d["images"], d["labels"]
n = im.shape[1]

def feats(w):
    u = np.triu(w, 1)
    return [w.mean(), w.max(), np.trace(w) / n,
            float(u[u > 0].mean()) if (u > 0).any() else 0]

names = ["mean", "max", "diag", "upper>0均值"]
F = np.array([feats(w) for w in im])
pos, neg = F[lab == 1], F[lab == 0]

def q(x):
    return np.percentile(x, [10, 25, 50, 75, 90]).round(3)

for j, nm in enumerate(names):
    print(f"\n[{nm}]")
    print("  正 p10-p90:", q(pos[:, j]), " 均值=%.3f" % pos[:, j].mean())
    print("  负 p10-p90:", q(neg[:, j]), " 均值=%.3f" % neg[:, j].mean())

# 负样本强度分布：有多少负样本 max 超过正样本中位数
neg_max = neg[:, 1]; pos_max_med = np.median(pos[:, 1])
print("\n负样本中 max 超过正样本中位数(%.2f) 的比例: %.0f%%" % (pos_max_med, 100 * (neg_max > pos_max_med).mean()))
print("负样本中 max 超过正样本 p25(%.2f) 的比例: %.0f%%" % (np.percentile(pos[:, 1], 25), 100 * (neg_max > np.percentile(pos[:, 1], 25)).mean()))
print("正样本 max 低于负样本中位数(%.2f) 的比例: %.0f%%" % (np.median(neg_max), 100 * (pos[:, 1] < np.median(neg_max)).mean()))
