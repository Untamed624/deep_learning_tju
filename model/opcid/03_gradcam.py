"""
给 OPCID 分类模型生成 Grad-CAM 显著性图。

作用：证明模型关注的是“结构本身”而不是背景密度（任务一的可解释性要求）。
输入统一降采样到 128×128 与训练一致；对最后一个卷积层做 Grad-CAM，
叠加到原始 O/E 窗口图上，输出 PNG。

运行：python 03_gradcam.py
依赖：torch, numpy, matplotlib
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE     = r"C:\Users\ZhuanZ1\Doubao\chats\2026-09-28\new-chat\opcid_task"
REP1_NPZ = BASE + r"\data_out\rep1_data.npz"
MODEL    = BASE + r"\data_out\opcid_cnn.pt"
OUT_DIR  = BASE + r"\data_out\gradcam"
SIZE     = 128
N_SHOW   = 6
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OPCIDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, 2))

    def forward(self, x):
        return self.fc(self.pool(self.features(x)))


def gradcam(model, x3d, target_cls):
    """x3d: (1,SIZE,SIZE)；返回热图 (SIZE,SIZE)。"""
    act = {}
    h = model.features[-1].register_forward_hook(lambda m, i, o: act.setdefault("a", o))
    x = x3d.unsqueeze(0).to(DEVICE)          # (1,1,SIZE,SIZE)
    logits = model(x)
    score = logits[0, target_cls]
    grad = torch.autograd.grad(score, act["a"])[0]
    a = act["a"][0].detach()                  # (C,h,w)
    g = grad[0].detach()
    w = g.mean(dim=(1, 2), keepdim=True)
    cam = (w * a).sum(dim=0).clamp(min=0).cpu().numpy()
    h.remove()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    f = SIZE // cam.shape[0]                  # 8×8 -> SIZE
    return np.kron(cam, np.ones((f, f)))


def main():
    d = np.load(REP1_NPZ, allow_pickle=False)
    images, labels = d["images"], d["labels"]
    model = OPCIDNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL, map_location=DEVICE))
    model.eval()
    os.makedirs(OUT_DIR, exist_ok=True)

    for target in [1, 0]:
        ids = np.where(labels == target)[0][:N_SHOW]
        for k, i in enumerate(ids):
            x0 = torch.from_numpy(images[i]).float().unsqueeze(0)         # (1,200,200)
            x0 = F.interpolate(x0.unsqueeze(0), size=(SIZE, SIZE),
                               mode="bilinear", align_corners=False).squeeze(0)  # (1,128,128)
            cam = gradcam(model, x0, target)
            img128 = x0[0].numpy()
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
            ax = axes[0]
            ax.imshow(img128, cmap="Reds", vmin=0, vmax=float(np.quantile(img128, 0.99)))
            ax.set_title("原始 O/E 窗口"); ax.axis("off")
            ax = axes[1]
            ax.imshow(img128, cmap="gray", vmin=0, vmax=float(np.quantile(img128, 0.99)))
            ax.imshow(cam, cmap="jet", alpha=0.55)
            ax.set_title("Grad-CAM 显著性"); ax.axis("off")
            name = f"gradcam_{'OPCID' if target == 1 else 'NON'}_{k + 1}.png"
            fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, name), dpi=120)
            plt.close(fig)
            print("saved:", os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    main()
