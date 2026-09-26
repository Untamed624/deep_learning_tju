"""Region-level Hi-C GNN training and whole-genome scanning.

Training examples are square genomic regions, not independently labelled bins.
Each annotated interval is retained at its original ``[Start, End]`` extent
for the label. The GNN view expands it by 500 bp, while the CNN uses three
contact-matrix views expanded by 250, 500, and 1000 bp. Only annotated regions
are used for training. Whole-genome scanning is only enabled with ``--scan``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from .GNN import GraphModel, MultiLabelClassifier, read_interval_labels
except ImportError:  # direct execution: python model/region_GNN.py
    from GNN import GraphModel, MultiLabelClassifier, read_interval_labels


class ContactCNN(nn.Module):
    """CNN for three-channel contact features and geometric patch views."""

    def __init__(self, output_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(64, output_dim), nn.GELU())

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(patch))

    def tokens(self, patch: torch.Tensor) -> torch.Tensor:
        # Return spatial CNN tokens for cross-attention instead of collapsing
        # the contact patch immediately.
        feature_map = self.features[:-1](patch)
        return feature_map.flatten(2).transpose(1, 2)


class CrossModalFusion(nn.Module):
    """Bidirectional attention between CNN contact tokens and GNN nodes."""

    def __init__(self, gnn_dim: int, cnn_token_dim: int, fusion_dim: int, heads: int = 4):
        super().__init__()
        self.gnn_projection = nn.Linear(gnn_dim, fusion_dim)
        self.cnn_projection = nn.Linear(cnn_token_dim, fusion_dim)
        self.cnn_queries_gnn = nn.MultiheadAttention(fusion_dim, heads, batch_first=True)
        self.gnn_queries_cnn = nn.MultiheadAttention(fusion_dim, heads, batch_first=True)
        self.norm_cnn = nn.LayerNorm(fusion_dim)
        self.norm_gnn = nn.LayerNorm(fusion_dim)

    def forward(self, gnn_nodes: torch.Tensor, cnn_tokens: torch.Tensor) -> torch.Tensor:
        gnn_tokens = self.gnn_projection(gnn_nodes).unsqueeze(0)
        cnn_tokens = self.cnn_projection(cnn_tokens).unsqueeze(0)
        cnn_attended, _ = self.cnn_queries_gnn(cnn_tokens, gnn_tokens, gnn_tokens)
        gnn_attended, _ = self.gnn_queries_cnn(gnn_tokens, cnn_tokens, cnn_tokens)
        cnn_summary = self.norm_cnn(cnn_tokens + cnn_attended).mean(dim=1).squeeze(0)
        gnn_summary = self.norm_gnn(gnn_tokens + gnn_attended).mean(dim=1).squeeze(0)
        return torch.cat((cnn_summary, gnn_summary))


class MultiScaleFusion(nn.Module):
    """Three scales and three geometric views followed by cross-attention."""

    def __init__(self, gnn_dim: int, fusion_dim: int = 64):
        super().__init__()
        self.branches = nn.ModuleList([ContactCNN(128) for _ in (250, 500, 1000)])
        self.attention = CrossModalFusion(gnn_dim, 64, fusion_dim)

    def forward(self, gnn_nodes: torch.Tensor, patches: list[torch.Tensor]) -> torch.Tensor:
        # Each item is [views=3, channels=3, H, W]. Treat geometric views as
        # additional contact tokens while keeping one CNN encoder per scale.
        tokens = []
        for branch, patch_views in zip(self.branches, patches):
            view_tokens = branch.tokens(patch_views)
            tokens.append(view_tokens.reshape(1, -1, view_tokens.size(-1)))
        return self.attention(gnn_nodes, torch.cat(tokens, dim=1).squeeze(0))


def build_regions(labels_dir: Path, genome_end: int, context: int, seed: int):
    """Build training regions from annotations without adding background samples.

    ``genome_end`` and ``seed`` remain accepted for compatibility with existing
    callers and saved command lines. Negative labels for individual classes are
    still represented by the multi-label target vector on annotated regions.
    """
    intervals = read_interval_labels(labels_dir)
    positives = []
    for _, start, end, label in intervals:
        left = max(0, start - context)
        right = min(genome_end, end + context)
        if right > left:
            positives.append((left, right, label))
    if not positives:
        raise ValueError("No positive regions found")

    label_index = {"OPCID": 0, "CHIN": 1, "CHID": 2}
    samples = []
    for start, end, label in positives:
        target = np.zeros(3, dtype=np.float32)
        target[label_index[label]] = 1.0
        # Preserve known overlaps, e.g. CHIN inside CHID.
        for other_start, other_end, other_label in positives:
            if other_label in label_index and start < other_end and end > other_start:
                target[label_index[other_label]] = 1.0
        samples.append((start, end, target))
    return samples


def region_graph(arrays, global_edge_index, global_edge_weight, start, end, device):
    node_mask = (arrays["node_start"] < end) & (arrays["node_end"] > start)
    nodes = np.flatnonzero(node_mask).astype(np.int64)
    if nodes.size == 0:
        return None
    local_map = np.full(arrays["node_start"].size, -1, dtype=np.int64)
    local_map[nodes] = np.arange(nodes.size, dtype=np.int64)
    source, target = global_edge_index
    keep = node_mask[source] & node_mask[target]
    local_edges = np.vstack((local_map[source[keep]], local_map[target[keep]])).astype(np.int64)
    x = torch.from_numpy(arrays["x"][nodes].astype(np.float32, copy=False)).to(device)
    edge_index = torch.from_numpy(local_edges).to(device)
    edge_weight = torch.from_numpy(global_edge_weight[keep].astype(np.float32, copy=False)).to(device)
    return x, edge_index, edge_weight, nodes


def region_embedding(model, graph):
    x, edge_index, edge_weight, _ = graph
    z = model(x, edge_index, edge_weight)
    topology = torch.log1p(torch.tensor([x.size(0), edge_index.size(1)], dtype=torch.float32, device=x.device))
    return torch.cat((z.mean(dim=0), topology))


def _contact_channels(raw: torch.Tensor) -> torch.Tensor:
    """Build log-contact, distance-standardized contact and diagonal-distance channels."""
    raw = torch.nan_to_num(raw.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    log_contact = torch.log1p(raw)
    scale = log_contact.amax().clamp_min(1e-6)
    log_contact = log_contact / scale

    n = log_contact.size(0)
    standardized = torch.zeros_like(log_contact)
    for offset in range(n):
        diagonal = torch.diagonal(log_contact, offset=offset)
        mean = diagonal.mean()
        std = diagonal.std(unbiased=False).clamp_min(1e-6)
        standardized.diagonal(offset=offset).copy_((diagonal - mean) / std)
        if offset:
            lower = torch.diagonal(log_contact, offset=-offset)
            lower_mean = lower.mean()
            lower_std = lower.std(unbiased=False).clamp_min(1e-6)
            standardized.diagonal(offset=-offset).copy_((lower - lower_mean) / lower_std)

    row = torch.arange(n, device=log_contact.device)[:, None]
    col = torch.arange(n, device=log_contact.device)[None, :]
    diagonal_distance = (row - col).abs().float() / max(n - 1, 1)
    return torch.stack((log_contact, standardized, diagonal_distance), dim=0)


def _diagonal_coordinate_view(channels: torch.Tensor) -> torch.Tensor:
    """Resample a patch in (mean genomic coordinate, diagonal offset) space."""
    _, n, _ = channels.shape
    coords = torch.linspace(-1.0, 1.0, n, device=channels.device)
    mean_coord, offset_coord = torch.meshgrid(coords, coords, indexing="ij")
    # u=(i+j)/2 and v=(i-j)/2, with output axes normalized to [-1, 1].
    i = (mean_coord + offset_coord).clamp(-1, 1)
    j = (mean_coord - offset_coord).clamp(-1, 1)
    grid = torch.stack((j, i), dim=-1).unsqueeze(0)
    return F.grid_sample(channels.unsqueeze(0), grid, mode="bilinear", padding_mode="border", align_corners=True).squeeze(0)


def _band_view(channels: torch.Tensor) -> torch.Tensor:
    """Unfold diagonals into rows: row=offset, column=position along diagonal."""
    _, n, _ = channels.shape
    band = torch.zeros((channels.size(0), 2 * n - 1, n), dtype=channels.dtype, device=channels.device)
    row = 0
    for offset in range(-(n - 1), n):
        values = torch.diagonal(channels, offset=offset, dim1=1, dim2=2)
        length = values.size(1)
        band[:, row, :length] = values
        row += 1
    return F.interpolate(band.unsqueeze(0), size=(n, n), mode="bilinear", align_corners=False).squeeze(0)


def contact_patch(cool_matrix, arrays, start, end, patch_size, device):
    """Extract three-channel features and three geometric views for one window."""
    node_mask = (arrays["node_start"] < end) & (arrays["node_end"] > start)
    nodes = np.flatnonzero(node_mask)
    if nodes.size == 0:
        return None
    # The graph bins and Cooler bins share their order and resolution.
    raw = np.asarray(cool_matrix[nodes[0] : nodes[-1] + 1, nodes[0] : nodes[-1] + 1], dtype=np.float32)
    channels = _contact_channels(torch.from_numpy(raw).to(device))
    views = [channels, _diagonal_coordinate_view(channels), _band_view(channels)]
    return torch.stack([
        F.interpolate(view.unsqueeze(0), size=(patch_size, patch_size), mode="bilinear", align_corners=False).squeeze(0)
        for view in views
    ])


def multiscale_patches(cool_matrix, arrays, center, contexts, patch_size, device):
    """Extract 250/500/1000 bp context windows around one region center."""
    patches = []
    genome_end = int(arrays["node_end"].max())
    for context in contexts:
        start = max(0, int(center[0] - context))
        end = min(genome_end, int(center[1] + context))
        patch = contact_patch(cool_matrix, arrays, start, end, patch_size, device)
        if patch is None:
            return None
        patches.append(patch)
    return patches


def graph_context_bounds(start, end, arrays, context=500):
    genome_end = int(arrays["node_end"].max())
    return max(0, int(start - context)), min(genome_end, int(end + context))


def fused_region_embedding(model, fusion, graph, patches):
    z = model(graph[0], graph[1], graph[2])
    topology = torch.log1p(torch.tensor([graph[0].size(0), graph[1].size(1)], dtype=torch.float32, device=z.device))
    return torch.cat((fusion(z, patches), topology))


def evaluate(classifier, embeddings, labels, device, threshold):
    with torch.no_grad():
        probability = torch.sigmoid(classifier(torch.stack(embeddings).to(device))).cpu().numpy()
    labels = np.asarray(labels, dtype=np.float32)
    prediction = (probability >= threshold).astype(np.float32)
    tp = (prediction * labels).sum(axis=0)
    precision = tp / np.maximum(prediction.sum(axis=0), 1.0)
    recall = tp / np.maximum(labels.sum(axis=0), 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {"binary_accuracy": float((prediction == labels).mean()), "precision": precision.tolist(), "recall": recall.tolist(), "f1": f1.tolist()}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--cool-input", type=Path, required=True, help="Original .cool file used for CNN matrix patches.")
    parser.add_argument("--labels", type=Path, default=root / "data/datasets")
    parser.add_argument("--output-dir", type=Path, default=root / "data/region_embeddings")
    parser.add_argument("--context", type=int, default=500, help="Recorded legacy value; GNN context is fixed at 500 bp and CNN contexts are 250/500/1000 bp.")
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--cnn-dim", type=int, default=128, help="Per-scale CNN token projection width (kept for compatibility).")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--classifier-hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--novelty-threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scan", action="store_true", help="After training, scan the whole chromosome.")
    parser.add_argument("--scan-window", type=int, default=0, help="Window size in bp; 0 uses the median positive region size.")
    parser.add_argument("--scan-step", type=int, default=500)
    args = parser.parse_args()
    if args.context < 0 or args.batch_size <= 0 or args.epochs <= 0:
        raise SystemExit("context, batch-size and epochs must be valid positive values")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    try:
        import cooler
    except ImportError as exc:
        raise SystemExit("CNN matrix input requires cooler; install it in the active environment.") from exc
    cool = cooler.Cooler(str(args.cool_input))
    cool_matrix = cool.matrix(balance=False, sparse=False)
    with np.load(args.input, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    genome_end = int(arrays["node_end"].max())
    # Keep the annotation intervals as the region labels. Context is applied
    # consistently below: 500 bp for the GNN view and 250/500/1000 bp for CNN.
    regions = build_regions(args.labels, genome_end, 0, args.seed)
    order = rng.permutation(len(regions))
    split = max(1, int(round(len(regions) * (1 - args.test_fraction))))
    train_regions = [regions[i] for i in order[:split]]
    test_regions = [regions[i] for i in order[split:]]
    edge_index = arrays["edge_index"]
    edge_weight = arrays["edge_weight"]
    model = GraphModel(arrays["x"].shape[1], args.hidden_dim, args.embedding_dim).to(device)
    cnn = MultiScaleFusion(args.embedding_dim, fusion_dim=args.embedding_dim).to(device)
    classifier = MultiLabelClassifier(args.embedding_dim * 2 + 2, args.classifier_hidden_dim).to(device)
    train_labels = np.stack([r[2] for r in train_regions])
    positive = torch.from_numpy(train_labels.sum(axis=0)).to(device).clamp_min(1.0)
    pos_weight = ((len(train_regions) - positive).clamp_min(1.0) / positive).float()
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(cnn.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=1e-5)

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_regions)
        model.train(); cnn.train(); classifier.train(); epoch_loss = 0.0
        for batch_start in range(0, len(train_regions), args.batch_size):
            batch = train_regions[batch_start:batch_start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for start, end, label in batch:
                graph_start, graph_end = graph_context_bounds(start, end, arrays, 500)
                graph = region_graph(arrays, edge_index, edge_weight, graph_start, graph_end, device)
                if graph is None:
                    continue
                patches = multiscale_patches(cool_matrix, arrays, (start, end), (250, 500, 1000), args.patch_size, device)
                if patches is None:
                    continue
                embedding = fused_region_embedding(model, cnn, graph, patches)
                losses.append(F.binary_cross_entropy_with_logits(classifier(embedding), torch.from_numpy(label).to(device), pos_weight=pos_weight))
            if losses:
                loss = torch.stack(losses).mean(); loss.backward(); optimizer.step(); epoch_loss += float(loss.detach().cpu())
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(f"region epoch {epoch:03d}/{args.epochs} loss={epoch_loss:.5f}")

    model.eval(); cnn.eval(); classifier.eval()
    train_emb, train_y, train_coords = [], [], []
    test_emb, test_y, test_coords = [], [], []
    with torch.no_grad():
        for collection, emb_out, y_out, coord_out in ((train_regions, train_emb, train_y, train_coords), (test_regions, test_emb, test_y, test_coords)):
            for start, end, label in collection:
                graph_start, graph_end = graph_context_bounds(start, end, arrays, 500)
                graph = region_graph(arrays, edge_index, edge_weight, graph_start, graph_end, device)
                if graph is not None:
                    patches = multiscale_patches(cool_matrix, arrays, (start, end), (250, 500, 1000), args.patch_size, device)
                    if patches is not None:
                        emb_out.append(fused_region_embedding(model, cnn, graph, patches).cpu())
                        y_out.append(label)
                        coord_out.append((start, end))
    if not train_emb or not test_emb:
        raise RuntimeError(
            f"No usable region embeddings were produced (train={len(train_emb)}, test={len(test_emb)}). "
            "Check that graph coordinates and .cool bins use the same chromosome/resolution."
        )
    stats = {"train": evaluate(classifier, train_emb, train_y, device, args.threshold), "test": evaluate(classifier, test_emb, test_y, device, args.threshold), "train_regions": len(train_emb), "test_regions": len(test_emb), "context": args.context}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    torch.save({"gnn_encoder": model.state_dict(), "cnn_encoder": cnn.state_dict(), "classifier": classifier.state_dict(), "args": vars(args)}, args.output_dir / f"{stem}_region_model.pt")
    np.savez_compressed(args.output_dir / f"{stem}_region_test.npz", embedding=torch.stack(test_emb).numpy(), labels=np.asarray(test_y), start=np.asarray([x[0] for x in test_coords]), end=np.asarray([x[1] for x in test_coords]), metadata=np.array(json.dumps(stats)))
    if args.scan:
        positive_lengths = [end - start for start, end, label in regions if label is not None]
        window = args.scan_window or int(np.median(positive_lengths))
        scan_rows = []
        with torch.no_grad():
            for start in range(0, max(1, genome_end - window + 1), args.scan_step):
                end = min(genome_end, start + window)
                graph_start, graph_end = graph_context_bounds(start, end, arrays, 500)
                graph = region_graph(arrays, edge_index, edge_weight, graph_start, graph_end, device)
                if graph is None:
                    continue
                patches = multiscale_patches(cool_matrix, arrays, (start, end), (250, 500, 1000), args.patch_size, device)
                if patches is None:
                    continue
                embedding = fused_region_embedding(model, cnn, graph, patches)
                probability = torch.sigmoid(classifier(embedding)).cpu().numpy()
                max_probability = float(probability.max())
                known = any(start < positive_end and end > positive_start for positive_start, positive_end, _ in regions)
                scan_rows.append((start, end, *probability.tolist(), 1.0 - max_probability, bool((not known) and max_probability < args.novelty_threshold)))
        scan_dtype = [("start", "i8"), ("end", "i8"), ("p_opcid", "f4"), ("p_chin", "f4"), ("p_chid", "f4"), ("novelty_score", "f4"), ("novel_candidate", "?")]
        np.save(args.output_dir / f"{stem}_whole_genome_scan.npy", np.asarray(scan_rows, dtype=scan_dtype))
        stats["scan_windows"] = len(scan_rows)
        stats["scan_novel_candidates"] = int(sum(row[-1] for row in scan_rows))
        print(json.dumps({"whole_genome_scan": stats["scan_windows"], "novel_candidates": stats["scan_novel_candidates"]}, ensure_ascii=False))
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
