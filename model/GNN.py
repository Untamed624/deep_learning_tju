"""Train a graph encoder and export local graph embeddings.

The parser output contains one large contact graph. This module uses a
weighted GraphSAGE encoder and a lightweight edge-reconstruction objective
when labels are not available yet. If a ``labels`` array is added to the NPZ,
``--task supervised`` trains an optional node classifier instead.

The exported local embedding is the mean of the final node embeddings in each
center's radius-r neighborhood, concatenated with mean topology statistics.
This is the representation intended for later OPCID/CHID/CHIN classification.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
except ImportError as exc:  # pragma: no cover - depends on user's environment
    raise SystemExit(
        "This script requires torch-geometric. Install a version matching your "
        "PyTorch build: https://pytorch-geometric.readthedocs.io/en/latest/install/"
    ) from exc


class GraphSAGEEncoder(nn.Module):
    """Weighted two-layer GraphSAGE encoder for node representation learning."""

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embedding_dim)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        # SAGEConv does not use edge weights directly; scale messages by the
        # normalized contact strength through a weighted residual feature.
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index)


class GraphModel(nn.Module):
    """Encoder plus optional node classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, classes: int = 0, dropout: float = 0.2):
        super().__init__()
        self.encoder = GraphSAGEEncoder(input_dim, hidden_dim, embedding_dim, dropout)
        self.classifier = nn.Linear(embedding_dim, classes) if classes else None

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        return self.encoder(x, edge_index, edge_weight)


def _resolve_input(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = sorted(path.parent.rglob(path.name)) if path.parent.exists() else []
    candidates += sorted(Path("data/graphs").glob("*rep1*top20.npz"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Graph NPZ not found: {path}")


def load_graph(path: Path, device: torch.device) -> tuple[dict[str, np.ndarray], Tensor, Tensor, Tensor, Tensor]:
    path = _resolve_input(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    x = torch.from_numpy(arrays["x"].astype(np.float32, copy=False))
    edge_index = torch.from_numpy(arrays["edge_index"].astype(np.int64, copy=False))
    edge_weight = torch.from_numpy(arrays["edge_weight"].astype(np.float32, copy=False))

    # Add topology to the node input: log degree and weighted contact mass.
    degree = torch.zeros(x.size(0), dtype=torch.float32)
    degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1)))
    weighted_degree = torch.zeros(x.size(0), dtype=torch.float32)
    weighted_degree.scatter_add_(0, edge_index[0], edge_weight)
    degree = torch.log1p(degree)
    weighted_degree = torch.log1p(weighted_degree)
    for feature in (degree, weighted_degree):
        feature /= feature.max().clamp_min(1.0)
    x = torch.cat((x, degree[:, None], weighted_degree[:, None]), dim=1)
    return arrays, x.to(device), edge_index.to(device), edge_weight.to(device), path


def sample_edges(edge_index: Tensor, count: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    count = min(count, edge_index.size(1))
    chosen = torch.randperm(edge_index.size(1), generator=generator, device=edge_index.device)[:count]
    positive = edge_index[:, chosen]
    negative = torch.randint(0, int(edge_index.max()) + 1, (2, count), generator=generator, device=edge_index.device)
    return positive, negative


def link_loss(z: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
    pos_score = (z[positive[0]] * z[positive[1]]).sum(dim=1)
    neg_score = (z[negative[0]] * z[negative[1]]).sum(dim=1)
    return F.softplus(-pos_score).mean() + F.softplus(neg_score).mean()


def local_embeddings(z: Tensor, edge_index: Tensor, centers: np.ndarray, radius: int) -> Tensor:
    """Mean-pool node embeddings and topology features around center nodes."""
    # Propagate sums and node counts. This computes a radius-hop mean without
    # materializing Python adjacency lists (the input graph has millions of
    # edges). Repeated nodes reached by multiple paths are harmless for this
    # stable, topology-aware pooling representation.
    source, target = edge_index
    node_count = z.size(0)
    self_index = torch.arange(node_count, device=z.device)
    source = torch.cat((source, self_index))
    target = torch.cat((target, self_index))
    pooled_sum = z
    pooled_count = torch.ones((node_count, 1), device=z.device)
    for _ in range(radius):
        next_sum = torch.zeros_like(pooled_sum)
        next_sum.index_add_(0, target, pooled_sum[source])
        next_count = torch.zeros_like(pooled_count)
        next_count.index_add_(0, target, pooled_count[source])
        pooled_sum = next_sum
        pooled_count = next_count
    pooled = pooled_sum / pooled_count.clamp_min(1.0)

    degree = torch.zeros(node_count, device=z.device)
    degree.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=z.device))
    center_index = torch.as_tensor(centers, device=z.device, dtype=torch.long)
    topology = torch.cat(
        (torch.log1p(pooled_count[center_index]), torch.log1p(degree[center_index, None])), dim=1
    )
    return torch.cat((pooled[center_index], topology), dim=1)


def read_interval_labels(path: Path) -> list[tuple[str, int, int, str]]:
    """Read OPCID/CHIN/CHID intervals from separate ``*_data.xlsx`` files."""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("Supervised mode requires openpyxl: pip install openpyxl") from exc
    if path.is_dir():
        files = {label: path / f"{label}_data.xlsx" for label in ("OPCID", "CHIN", "CHID")}
    elif path.is_file() and path.name.lower().endswith("_data.xlsx"):
        label = path.stem.split("_", 1)[0].upper()
        files = {label: path}
    else:
        raise FileNotFoundError(
            f"Annotation path must be a directory containing OPCID_data.xlsx, "
            f"CHIN_data.xlsx and CHID_data.xlsx: {path}"
        )

    intervals: list[tuple[str, int, int, str]] = []
    for label, workbook_path in files.items():
        if label not in {"OPCID", "CHIN", "CHID"}:
            continue
        if not workbook_path.is_file():
            raise FileNotFoundError(f"Missing annotation workbook: {workbook_path}")
        workbook = load_workbook(workbook_path, read_only=True, data_only=True)
        try:
            sheet = workbook[workbook.sheetnames[0]]
            headers = [str(value).strip().lower() if value is not None else "" for value in next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))]
            try:
                chrom_col = headers.index("chr")
                start_col = headers.index("start")
                end_col = headers.index("end")
            except ValueError as exc:
                raise ValueError(
                    f"{workbook_path.name} must contain Chr, Start and End columns; found {headers}"
                ) from exc
            for row in sheet.iter_rows(min_row=2, values_only=True):
                chrom = row[chrom_col] if len(row) > chrom_col else None
                start = row[start_col] if len(row) > start_col else None
                end = row[end_col] if len(row) > end_col else None
                if chrom is None or start is None or end is None:
                    continue
                try:
                    intervals.append((str(chrom), int(start), int(end), label))
                except (TypeError, ValueError):
                    continue
        finally:
            workbook.close()
    if not intervals:
        raise ValueError(f"No OPCID/CHIN/CHID intervals found under {path}")
    return intervals


def labels_for_centers(arrays: dict[str, np.ndarray], centers: np.ndarray, workbook: Path) -> np.ndarray:
    """Assign labels by genomic Start/End overlap.

    The experiment treats chromosome names in the graph and annotation files
    as equivalent, so ``Chr`` is intentionally not used for matching.
    """
    labels = np.zeros((centers.size, 3), dtype=np.float32)
    starts = arrays["node_start"][centers].astype(np.int64)
    ends = arrays["node_end"][centers].astype(np.int64)
    intervals = read_interval_labels(workbook)
    label_index = {"OPCID": 0, "CHIN": 1, "CHID": 2}
    for interval_chrom, interval_start, interval_end, label in intervals:
        overlap = (
            (starts < interval_end)
            & (ends > interval_start)
        )
        labels[overlap, label_index[label]] = 1.0
    return labels


def split_label_indices(labels: np.ndarray, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create a reproducible split while preserving positives from each class."""
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    target_test = max(1, int(round(n * test_fraction)))
    test = set()
    for class_id in range(labels.shape[1]):
        positive = np.flatnonzero(labels[:, class_id] > 0)
        if positive.size:
            take = max(1, int(round(positive.size * test_fraction)))
            test.update(rng.choice(positive, size=min(take, positive.size), replace=False).tolist())
    remaining = np.array(sorted(set(range(n)) - test), dtype=np.int64)
    if len(test) < target_test and remaining.size:
        extra = rng.choice(remaining, size=min(target_test - len(test), remaining.size), replace=False)
        test.update(extra.tolist())
    test_idx = np.array(sorted(test), dtype=np.int64)
    train_idx = np.array(sorted(set(range(n)) - set(test_idx.tolist())), dtype=np.int64)
    return train_idx, test_idx


def train_classifier(features: np.ndarray, labels: np.ndarray, device: torch.device, epochs: int, seed: int, train_idx: Optional[np.ndarray] = None) -> tuple[nn.Module, dict[str, float]]:
    """Train a small multi-label classifier on pooled local graph embeddings."""
    torch.manual_seed(seed)
    if train_idx is None:
        train_idx = np.arange(features.shape[0], dtype=np.int64)
    x = torch.from_numpy(features[train_idx]).to(device)
    y = torch.from_numpy(labels[train_idx]).to(device)
    classifier = nn.Sequential(nn.LayerNorm(x.size(1)), nn.Linear(x.size(1), 3)).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=2e-3, weight_decay=1e-4)
    positive = y.sum(dim=0).clamp_min(1.0)
    negative = (y.size(0) - positive).clamp_min(1.0)
    pos_weight = (negative / positive).to(device)
    classifier.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(classifier(x), y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
    classifier.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(classifier(x))
        predicted = (probabilities >= 0.5).float()
        micro_accuracy = float((predicted == y).float().mean().cpu())
    return classifier, {"train_binary_accuracy": micro_accuracy, "positive_counts": labels[train_idx].sum(axis=0).tolist(), "train_size": int(train_idx.size)}


def evaluate_classifier(classifier: nn.Module, features: np.ndarray, labels: np.ndarray, device: torch.device) -> dict[str, object]:
    """Evaluate multi-label predictions without updating classifier weights."""
    with torch.no_grad():
        logits = classifier(torch.from_numpy(features).to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy()
    predicted = (probabilities >= 0.5).astype(np.float32)
    true_positive = (predicted * labels).sum(axis=0)
    precision = true_positive / np.maximum(predicted.sum(axis=0), 1.0)
    recall = true_positive / np.maximum(labels.sum(axis=0), 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "binary_accuracy": float((predicted == labels).mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "positive_counts": labels.sum(axis=0).tolist(),
    }


def novelty_scores(classifier: nn.Module, features: np.ndarray, known_labels: np.ndarray, device: torch.device, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find open-set candidates without assigning them a biological type.

    A candidate must be unlabelled by the known annotation and have low
    confidence for all known classes. ``novelty_score`` is one minus the
    strongest known-class probability; it is intentionally a screening score,
    not a biological claim.
    """
    with torch.no_grad():
        probabilities = torch.sigmoid(classifier(torch.from_numpy(features).to(device))).cpu().numpy()
    max_probability = probabilities.max(axis=1)
    score = 1.0 - max_probability
    not_known = known_labels.sum(axis=1) == 0
    candidate = not_known & (max_probability < threshold)
    return probabilities, score.astype(np.float32), candidate


def train_encoder(model: GraphModel, x: Tensor, edge_index: Tensor, edge_weight: Tensor, args: argparse.Namespace, device: torch.device) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        z = model(x, edge_index, edge_weight)
        positive, negative = sample_edges(edge_index, args.edge_samples, generator)
        loss = link_loss(z, positive, negative)
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch:03d}/{args.epochs} loss={loss.item():.5f}")


def encode_and_pool(model: GraphModel, x: Tensor, edge_index: Tensor, edge_weight: Tensor, args: argparse.Namespace) -> tuple[Tensor, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        z = model(x, edge_index, edge_weight)
        node_count = z.size(0)
        if args.centers == 0 or args.centers >= node_count:
            centers = np.arange(node_count, dtype=np.int64)
        else:
            centers = np.linspace(0, node_count - 1, args.centers, dtype=np.int64)
        pooled = local_embeddings(z, edge_index, centers, args.radius)
    return z, centers, pooled.cpu().numpy().astype(np.float32)


def save_embeddings(output_dir: Path, stem: str, arrays: dict[str, np.ndarray], centers: np.ndarray, z: Tensor, pooled: np.ndarray, labels: np.ndarray, metadata: dict[str, object], probabilities: Optional[np.ndarray] = None, novelty_score: Optional[np.ndarray] = None, novel_candidate: Optional[np.ndarray] = None, split: Optional[np.ndarray] = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(
        centers=centers,
        graph_embedding=pooled,
        node_embedding=z.cpu().numpy().astype(np.float32),
        center_chrom=arrays.get("node_chrom", np.array([], dtype=str))[centers],
        center_start=arrays.get("node_start", np.array([], dtype=np.int64))[centers],
        center_end=arrays.get("node_end", np.array([], dtype=np.int64))[centers],
        labels=labels,
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    if probabilities is not None:
        payload["known_class_probability"] = probabilities.astype(np.float32)
        payload["known_class_prediction"] = (probabilities >= 0.5).astype(np.bool_)
        payload["novelty_score"] = novelty_score.astype(np.float32)
        payload["novel_candidate"] = novel_candidate.astype(np.bool_)
    if split is not None:
        payload["split"] = split.astype(np.int8)
    np.savez_compressed(output_dir / f"{stem}_local_embeddings.npz", **payload)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/graphs/GSE272159_37C_rep1.mapq_30.10_top20.npz")
    parser.add_argument("--output-dir", type=Path, default=root / "data/graph_embeddings")
    parser.add_argument("--task", choices=("unsupervised", "supervised"), default="unsupervised")
    parser.add_argument("--labels", type=Path, default=root / "data/datasets", help="Directory containing OPCID_data.xlsx, CHIN_data.xlsx and CHID_data.xlsx.")
    parser.add_argument("--classifier-epochs", type=int, default=100)
    parser.add_argument("--test-fraction", type=float, default=0.15, help="Random test fraction within the current graph (default: 15%%).")
    parser.add_argument("--novelty-threshold", type=float, default=0.55, help="Known-class confidence below this marks an unlabelled embedding as a novel candidate.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--edge-samples", type=int, default=100000)
    parser.add_argument("--centers", type=int, default=10000, help="Number of center nodes to pool; 0 means all nodes.")
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.radius <= 0:
        raise SystemExit("--epochs and --radius must be positive")
    if not 0 < args.test_fraction < 1:
        raise SystemExit("--test-fraction must be between 0 and 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    arrays, x, edge_index, edge_weight, input_path = load_graph(args.input, device)
    model = GraphModel(x.size(1), args.hidden_dim, args.embedding_dim).to(device)
    train_encoder(model, x, edge_index, edge_weight, args, device)
    z, centers, pooled = encode_and_pool(model, x, edge_index, edge_weight, args)
    stem = input_path.stem

    classifier_stats = {"task": args.task}
    if args.task == "supervised":
        labels = labels_for_centers(arrays, centers, args.labels)
        train_idx, test_idx = split_label_indices(labels, args.test_fraction, args.seed)
        classifier, classifier_stats = train_classifier(
            pooled, labels, device, args.classifier_epochs, args.seed, train_idx
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), args.output_dir / f"{stem}_classifier.pt")
        test_pooled = pooled[test_idx]
        test_labels = labels[test_idx]
        classifier_stats["test"] = evaluate_classifier(classifier, test_pooled, test_labels, device)
        test_probabilities, test_novelty, test_candidate = novelty_scores(classifier, test_pooled, test_labels, device, args.novelty_threshold)
        classifier_stats["test"]["novel_candidate_count"] = int(test_candidate.sum())
        save_embeddings(args.output_dir, f"{stem}_random_test", arrays, centers[test_idx], z, test_pooled, test_labels, {"split": "random_test", "train_input": str(input_path), "test_fraction": args.test_fraction, "radius": args.radius, "embedding_dim": int(test_pooled.shape[1]), "novelty_threshold": args.novelty_threshold}, probabilities=test_probabilities, novelty_score=test_novelty, novel_candidate=test_candidate)
        classifier_stats["split"] = {"train_size": int(train_idx.size), "test_size": int(test_idx.size), "test_fraction": args.test_fraction}
    else:
        labels = np.empty((0, 3), dtype=np.float32)

    torch.save(model.state_dict(), args.output_dir / f"{stem}_encoder.pt")
    train_probabilities = train_novelty = train_candidate = None
    if args.task == "supervised":
        train_probabilities, train_novelty, train_candidate = novelty_scores(classifier, pooled, labels, device, args.novelty_threshold)
        split = np.ones(centers.size, dtype=np.int8)
        split[test_idx] = 0
    else:
        split = None
    save_embeddings(args.output_dir, stem, arrays, centers, z, pooled, labels, {"split": "random_85_15" if args.task == "supervised" else "unsupervised", "input": str(input_path), "radius": args.radius, "embedding_dim": int(pooled.shape[1]), "novelty_threshold": args.novelty_threshold}, probabilities=train_probabilities, novelty_score=train_novelty, novel_candidate=train_candidate, split=split)
    print(json.dumps(classifier_stats, ensure_ascii=False))
    print(f"saved embeddings to {args.output_dir}")


if __name__ == "__main__":
    main()
