"""直接查 npz 里的窗口，看正样本信号和坐标。"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np

BASE = r"C:\Users\ZhuanZ1\Doubao\chats\2026-09-28\new-chat\opcid_task"
d = np.load(BASE + r"\data_out\rep1_data.npz", allow_pickle=False)
im, lab, coords = d["images"], d["labels"], d["coords"]
print("images shape:", im.shape)

pos_ids = np.where(lab == 1)[0]
neg_ids = np.where(lab == 0)[0]
print("正样本数:", len(pos_ids), " 负样本数:", len(neg_ids))

# 正样本 max 分布
pos_max = im[pos_ids].reshape(len(pos_ids), -1).max(1)
neg_max = im[neg_ids].reshape(len(neg_ids), -1).max(1)
print("\n正样本窗口 max: p25=%.3f p50=%.3f p75=%.3f  >0的个数=%d/%d"
      % (np.percentile(pos_max, 25), np.percentile(pos_max, 50), np.percentile(pos_max, 75),
         (pos_max > 0).sum(), len(pos_max)))
print("负样本窗口 max: p25=%.3f p50=%.3f p75=%.3f  >0的个数=%d/%d"
      % (np.percentile(neg_max, 25), np.percentile(neg_max, 50), np.percentile(neg_max, 75),
         (neg_max > 0).sum(), len(neg_max)))

# 前8 vs 后8 正样本的信号
print("\n正样本窗口信号（前8个 / 后8个）：")
for label, ids in [("前8", pos_ids[:8]), ("后8", pos_ids[-8:])]:
    print(label, "-> max:", [round(float(im[i].max()), 2) for i in ids])
    print("       sum:", [round(float(im[i].sum()), 1) for i in ids])
    print("       coord:", [f"{coords[i,0]/1000:.1f}-{coords[i,1]/1000:.1f}" for i in ids])

# 找出完全没有接触的正样本（可能切窗有问题）
zero_pos = pos_ids[pos_max == 0]
print("\n正样本中窗口全0(无接触)的个数:", len(zero_pos), "/", len(pos_ids))
if len(zero_pos):
    print("  这些坐标:", [f"{coords[i,0]}-{coords[i,1]}" for i in zero_pos[:10]])
