"""画几个切出的窗口，看 OPCID 矩形是否可见、负样本像不像背景。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE = r"C:\Users\ZhuanZ1\Doubao\chats\2026-09-28\new-chat\opcid_task"
d = np.load(BASE + r"\data_out\rep1_data.npz", allow_pickle=False)
images, labels, coords = d["images"], d["labels"], d["coords"]

pos_ids = np.where(labels == 1)[0][:6]
neg_ids = np.where(labels == 0)[0][:6]

fig, axes = plt.subplots(2, 6, figsize=(16, 5.5))
for j, i in enumerate(pos_ids):
    ax = axes[0][j]
    ax.imshow(images[i], cmap="Reds", vmin=0, vmax=float(np.quantile(images[i], 0.99)))
    ax.set_title(f"OPCID #{j+1}  {coords[i,0]/1000:.1f}-{coords[i,1]/1000:.1f} kb")
    ax.axis("off")
for j, i in enumerate(neg_ids):
    ax = axes[1][j]
    ax.imshow(images[i], cmap="Reds", vmin=0, vmax=float(np.quantile(images[i], 0.99)))
    ax.set_title(f"NON #{j+1}  {coords[i,0]/1000:.1f}-{coords[i,1]/1000:.1f} kb")
    ax.axis("off")
fig.suptitle("rep1 切窗预览：上排=OPCID正样本，下排=非OPCID负样本（O/E归一化）")
fig.tight_layout()
out = BASE + r"\data_out\preview_rep1.png"
fig.savefig(out, dpi=130)
print("saved:", out)
