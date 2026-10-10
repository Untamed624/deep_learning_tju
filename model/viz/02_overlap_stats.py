#调试文件，无实质用处
import openpyxl

def load_regions(path, id_col):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    regions = []
    header = None
    for row in ws.iter_rows(values_only=True):
        if header is None:
            header = [str(c).lower() if c else "" for c in row]
            continue
        rid = start = end = None
        for i, val in enumerate(row):
            col = header[i] if i < len(header) else ""
            if id_col in col:
                rid = str(val)
            elif col in ("start", "begin"):
                start = val
            elif col in ("end", "stop"):
                end = val
        if start is None or end is None:
            continue
        try:
            regions.append((rid, int(start), int(end)))
        except (ValueError, TypeError):
            continue
    wb.close()
    return regions

opcid = load_regions("data/datasets/OPCID_data.xlsx", "opcid")
chin = load_regions("data/datasets/CHIN_data.xlsx", "chin")
chid = load_regions("data/datasets/CHID_data.xlsx", "chid")

print(f"OPCID: {len(opcid)} 个")
print(f"CHIN:  {len(chin)} 个")
print(f"CHID:  {len(chid)} 个")
print()

def overlap_count(a_list, b_list):
    """统计 a_list 中有多少个区间和 b_list 重叠"""
    count = 0
    for a_id, a_s, a_e in a_list:
        for b_id, b_s, b_e in b_list:
            if a_s < b_e and b_s < a_e:
                count += 1
                break
    return count

print("=== 实际重叠统计 ===")
print(f"CHID 中有多少落在 CHIN 范围内: {overlap_count(chid, chin)} / {len(chid)}")
print(f"CHIN 中有多少落在 CHID 范围内: {overlap_count(chin, chid)} / {len(chin)}")
print()
print(f"OPCID 中有多少和 CHIN 重叠: {overlap_count(opcid, chin)} / {len(opcid)}")
print(f"OPCID 中有多少和 CHID 重叠: {overlap_count(opcid, chid)} / {len(opcid)}")
print(f"CHIN 中有多少和 OPCID 重叠: {overlap_count(chin, opcid)} / {len(chin)}")
print(f"CHID 中有多少和 OPCID 重叠: {overlap_count(chid, opcid)} / {len(chid)}")
print()

# 看几个 CHID 的例子
print("=== 前 5 个 CHID 样本 ===")
for rid, s, e in chid[:5]:
    print(f"  {rid}: {s}-{e} (长度 {e-s})")

print()
print("=== CHID 列名 ===")
wb = openpyxl.load_workbook("data/datasets/CHID_data.xlsx", read_only=True)
ws = wb.active
for row in ws.iter_rows(values_only=True):
    print(" ", row)
    break
wb.close()
