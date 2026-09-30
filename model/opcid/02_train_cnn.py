"""
训练 OPCID 二分类 CNN（正负样本 1:1）。

数据来源：01_build_dataset.py 生成的 data_out/rep1_data.npz（训练）与 rep2_data.npz（跨重复验证）。
流程：
  1. 读 rep1 数据，按 85/15 分层划分 train / val（固定种子）；
  2. 小 CNN + 数据增强（旋转/翻转/噪声/对比度扰动）训练；
  3. 早停按 val 准确率，保存最优权重；
  4. 在 rep2（另一个生物学重复）上做跨重复测试——这是作业加分项；
  5. 输出：指标数值表、混淆矩阵图、训练曲线、权重文件。

运行：python 02_train_cnn.py
依赖：torch, numpy, matplotlib（均已就绪）
"""
from __future__ import annotations
import os, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows OpenMP 冲突绕过
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# ====================== 配置区 ======================
BASE      = r"C:\Users\ZhuanZ1\Doubao\chats\2026-09-28\new-chat\opcid_task"
REP1_NPZ  = BASE + r"\data_out\rep1_data.npz"
REP2_NPZ  = BASE + r"\data_out\rep2_data.npz"
OUT_DIR   = BASE + r"\data_out"

SEED     = 42
TRAIN_RATIO = 0.85
EPOCHS   = 50
BATCH    = 16
LR       = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 20          # 早停耐心（val 样本少，放宽避免过早停）
SIZE     = 128         # 训练/测试输入边长（从 200 降采样，提升小样本学习效率）
# ====================================================

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OPCIDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)          # 全局平均池化 -> (128)
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, 2))

    def forward(self, x):
        return self.fc(self.pool(self.features(x)))


class AugDataset(Dataset):
    """带在线数据增强的 OPCID 窗口数据集。img: (200,200) float32"""
    def __init__(self, images, labels, train=True):
        self.images, self.labels, self.train = images, labels, train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).float().unsqueeze(0)               # (1,H,W) = (C,H,W)
        if img.shape[-1] != SIZE:
            img = F.interpolate(img.unsqueeze(0), size=(SIZE, SIZE),
                                mode="bilinear", align_corners=False).squeeze(0)
        y = self.labels[i]
        if self.train:
            if np.random.rand() > 0.5: img = img.flip(2)   # 水平翻转(W)，保持对角线结构
            if np.random.rand() > 0.5: img = img.flip(1)   # 垂直翻转(H)
            if np.random.rand() > 0.5:
                img = img + torch.randn_like(img) * 0.02
            if np.random.rand() > 0.5:                       # 对比度扰动
                img = img * float(np.random.uniform(0.85, 1.15))
        return img, y


def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return d["images"], d["labels"], d["coords"]


def metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    images1, labels1, coords1 = load_npz(REP1_NPZ)
    images2, labels2, coords2 = load_npz(REP2_NPZ)
    print(f"rep1: {images1.shape}  正={int((labels1==1).sum())} 负={int((labels1==0).sum())}")
    print(f"rep2: {images2.shape}  正={int((labels2==1).sum())} 负={int((labels2==0).sum())}")

    # ---- 分层划分 train/val（按正负比例分别切）----
    idx_pos = np.where(labels1 == 1)[0]
    idx_neg = np.where(labels1 == 0)[0]
    def split(ids):
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(ids))
        n_tr = int(len(ids) * TRAIN_RATIO)
        return ids[perm[:n_tr]], ids[perm[n_tr:]]
    tr_p, va_p = split(idx_pos)
    tr_n, va_n = split(idx_neg)
    tr_idx = np.concatenate([tr_p, tr_n]); va_idx = np.concatenate([va_p, va_n])
    np.random.shuffle(tr_idx)

    # 离线扩增训练集：原图 + 上下翻 + 左右翻 + 180翻 = 4 倍（只做保持对角线结构的变换）
    _exp_imgs, _exp_lab = [], []
    for _i in tr_idx:
        a = images1[_i]
        _exp_imgs += [a, np.flip(a, 0), np.flip(a, 1), np.flip(np.flip(a, 0), 1)]
        _exp_lab += [labels1[_i]] * 4
    train_ds = AugDataset(np.stack(_exp_imgs), np.array(_exp_lab), train=True)
    val_ds   = AugDataset(images1[va_idx], labels1[va_idx], train=False)
    test_ds  = AugDataset(images2, labels2, train=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    test_dl  = DataLoader(test_ds, batch_size=BATCH, shuffle=False)
    print(f"train={len(tr_idx)}  val={len(va_idx)}  test(rep2)={len(test_ds)}")

    model = OPCIDNet().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    # ---- 手写 Adam（torch.optim 初始化会触发 torch._dynamo/inductor，其依赖 triton.backends 缺失，故绕开）----
    beta1, beta2, eps, wd = 0.9, 0.999, 1e-8, WEIGHT_DECAY
    mstate = {p: torch.zeros_like(p) for p in model.parameters()}
    vstate = {p: torch.zeros_like(p) for p in model.parameters()}
    step = 0
    lr = LR

    best_val, best_state, no_improve = -1.0, None, 0
    hist = {"loss": [], "val_acc": []}

    def evaluate(dl):
        model.eval()
        ys, probs = [], []
        with torch.no_grad():
            for xb, yb in dl:
                xb = xb.to(DEVICE)
                p = torch.softmax(model(xb), 1)
                ys.append(yb.numpy()); probs.append(p[:, 1].cpu().numpy())
        return np.concatenate(ys), np.concatenate(probs)

    for ep in range(1, EPOCHS + 1):
        model.train()
        tot_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            step += 1
            with torch.no_grad():
                for p in model.parameters():
                    g = p.grad
                    if g is None:
                        continue
                    if wd:
                        g = g + wd * p
                    m = mstate[p]; m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v = vstate[p]; v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    mb = m / (1 - beta1 ** step)
                    vb = v / (1 - beta2 ** step)
                    p.addcdiv_(mb, vb.sqrt().add_(eps), value=-lr)
            tot_loss += loss.item() * len(xb)
        yv, pv1 = evaluate(val_dl)
        va = accuracy_score(yv, (pv1 >= 0.5).astype(int))
        hist["loss"].append(tot_loss / len(train_ds)); hist["val_acc"].append(va)
        # 手写 lr 调度：连续 6 次无提升则学习率减半
        if no_improve and no_improve % 6 == 0:
            lr *= 0.5
        if va > best_val:
            best_val, no_improve = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        print(f"ep {ep:3d}/{EPOCHS}  loss={hist['loss'][-1]:.4f}  val_acc={va:.4f}  best={best_val:.4f}", flush=True)
        if no_improve >= PATIENCE:
            print(f"early stop @ {ep}")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(OUT_DIR, "opcid_cnn.pt"))
    print("saved:", os.path.join(OUT_DIR, "opcid_cnn.pt"))

    # ---- 最终评估：val(rep1留出) + test(rep2跨重复) ----
    yv, pv1 = evaluate(val_dl)
    yt, pt1 = evaluate(test_dl)
    # 在 val 上扫描决策阈值，选 F1 最优者（平衡召回与误报）
    best_th, best_f1 = 0.5, -1.0
    for th in np.arange(0.5, 0.96, 0.05):
        f1v = f1_score(yv, (pv1 >= th).astype(int), zero_division=0)
        if f1v > best_f1:
            best_f1, best_th = f1v, th
    pv = (pv1 >= best_th).astype(int)
    pt = (pt1 >= best_th).astype(int)
    print(f"最优决策阈值 = {best_th:.2f} (val F1={best_f1:.3f})")
    m_val = metrics(yv, pv); m_test = metrics(yt, pt)

    for name, m in [("val (rep1留出15%)", m_val), ("test (rep2跨重复)", m_test)]:
        print(f"[{name}] acc={m['acc']:.3f} prec={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")

    # ---- 图：混淆矩阵 + 训练曲线 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (yt_mat, pt_mat, title) in zip(axes[:2], [(yv, pv, "混淆矩阵 · val(rep1留出)"),
                                                      (yt, pt, "混淆矩阵 · test(rep2跨重复)")]):
        cm = confusion_matrix(yt_mat, pt_mat)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title); ax.set_xlabel("预测"); ax.set_ylabel("真实")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["非OPCID", "OPCID"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["非OPCID", "OPCID"])
        for r in range(2):
            for c_ in range(2):
                ax.text(c_, r, int(cm[r, c_]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)
    ax = axes[2]
    ax.plot(hist["loss"], label="train loss"); ax.plot(hist["val_acc"], label="val acc")
    ax.set_title("训练曲线"); ax.set_xlabel("epoch"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "opcid_metrics.png"), dpi=150)
    print("saved:", os.path.join(OUT_DIR, "opcid_metrics.png"))

    # ---- 指标表 txt ----
    with open(os.path.join(OUT_DIR, "opcid_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("指标 | acc | precision | recall | f1\n")
        for name, m in [("val(rep1留出15%)", m_val), ("test(rep2跨重复)", m_test)]:
            f.write(f"{name} | {m['acc']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f}\n")
        f.write("\n混淆矩阵 val(rep1留出):\n" + str(confusion_matrix(yv, pv)) + "\n")
        f.write("混淆矩阵 test(rep2跨重复):\n" + str(confusion_matrix(yt, pt)) + "\n")
    print("saved:", os.path.join(OUT_DIR, "opcid_metrics.txt"))


if __name__ == "__main__":
    main()
