"""查正负样本能不能分开：逻辑回归 AUC + 特征分布。"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np

BASE = r"C:\Users\ZhuanZ1\Doubao\chats\2026-09-28\new-chat\opcid_task"
d = np.load(BASE + r"\data_out\rep1_data.npz", allow_pickle=False)
im, lab = d["images"], d["labels"]

print("样本数:", len(lab), " 正:", int((lab==1).sum()), " 负:", int((lab==0).sum()))

# ---- 统计特征 ----
n = im.shape[1]
feats = []
for i in range(len(im)):
    w = im[i]
    upper = np.triu(w, 1)
    feats.append([
        w.mean(),                                  # 窗口总体水平
        w.max(),
        np.percentile(w, 99.5),
        np.std(w),
        np.trace(w) / n,                           # 对角线强度
        np.mean(upper[upper > 0]) if (upper > 0).any() else 0,  # 正接触均值
    ])
feats = np.array(feats)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.decomposition import PCA
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    auc_stat = cross_val_score(LogisticRegression(max_iter=1000), feats, lab, cv=cv, scoring="roc_auc")
    print("统计特征 + 逻辑回归  AUC = %.3f (+/- %.3f)" % (auc_stat.mean(), auc_stat.std()))

    X = im.reshape(len(im), -1)
    P = PCA(n_components=20, random_state=0).fit_transform(X)
    auc_pca = cross_val_score(LogisticRegression(max_iter=2000), P, lab, cv=cv, scoring="roc_auc")
    print("PCA(20) + 逻辑回归  AUC = %.3f (+/- %.3f)" % (auc_pca.mean(), auc_pca.std()))
except Exception as e:
    print("sklearn 不可用:", repr(e))

# ---- 正负特征均值对比 ----
print("\n特征均值对比（正 / 负）:")
names = ["mean", "max", "p99.5", "std", "diag", "upper>0均值"]
for j, nm in enumerate(names):
    print(f"  {nm:<10} 正={feats[lab==1][:,j].mean():.4f}  负={feats[lab==0][:,j].mean():.4f}")
