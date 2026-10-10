#调试文件，无实质用处
genes = []
with open('data/datasets/NC_000913.3.gff', encoding='utf-8') as f:
    for line in f:
        if line.startswith('#'):
            continue
        parts = line.strip().split('\t')
        if len(parts) < 9:
            continue
        seqname, source, feature, start, end, score, strand, frame, attrs = parts
        if seqname != 'NC_000913.3' or feature != 'gene':
            continue
        s, e = int(start), int(end)
        if e < 49000 or s > 61000:
            continue
        name = ''
        for attr in attrs.split(';'):
            if attr.startswith('Name='):
                name = attr.split('=')[1]
                break
        genes.append((s, e, e - s, name))

genes.sort()
print(f"{'Start':>8} {'End':>8} {'Len':>6}  Name")
print('-' * 45)
for s, e, l, n in genes:
    flag = ' <-- 标了名' if l > 800 else ' <-- 太短没标'
    print(f"{s:>8} {e:>8} {l:>6}  {n}{flag}")
