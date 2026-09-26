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

class EdgeWeightedMessagePassing(nn.Module):
    """Mean GraphSAGE-style aggregation with normalized Hi-C edge weights."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbor_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.norm = nn.LayerNorm(output_dim)
        self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        source, target = edge_index
        # Log compression prevents a few high-count contacts from dominating.
        weight = torch.log1p(edge_weight.float()).clamp_min(0.0)
        denominator = torch.zeros(x.size(0), device=x.device, dtype=weight.dtype)
        denominator.index_add_(0, target, weight)
        normalized_weight = weight / denominator[target].clamp_min(1e-12)
        messages = self.neighbor_linear(x[source]) * normalized_weight.unsqueeze(1)
        aggregated = torch.zeros((x.size(0), messages.size(1)), device=x.device, dtype=messages.dtype)
        aggregated.index_add_(0, target, messages)
        # The residual/self term preserves the center bin's own features.
        return self.norm(self.self_linear(x) + aggregated + self.residual(x))


class GraphSAGEEncoder(nn.Module):
    """Two-layer edge-weighted message-passing encoder for Hi-C graphs."""

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.conv1 = EdgeWeightedMessagePassing(input_dim, hidden_dim)
        self.conv2 = EdgeWeightedMessagePassing(hidden_dim, embedding_dim)
        self.dropout = dropout

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        h = F.gelu(self.conv1(x, edge_index, edge_weight))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.conv2(h, edge_index, edge_weight)


class GraphModel(nn.Module):
    """Encoder plus optional node classifier."""

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, classes: int = 0, dropout: float = 0.2):
        super().__init__()
        self.encoder = GraphSAGEEncoder(input_dim, hidden_dim, embedding_dim, dropout)
        self.classifier = nn.Linear(embedding_dim, classes) if classes else None

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor) -> Tensor:
        return self.encoder(x, edge_index, edge_weight)


class MultiLabelClassifier(nn.Module):
    """Non-linear multi-label head; sigmoid is applied only for inference."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.25):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


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
    classifier = MultiLabelClassifier(x.size(1)).to(device)
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


class CSRNeighborSampler:
    """CPU CSR sampler for local contact subgraphs."""

    def __init__(self, edge_index: Tensor, edge_weight: Tensor, seed: int):
        source = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
        target = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
        weight = edge_weight.detach().cpu().numpy().astype(np.float32, copy=False)
        order = np.argsort(source, kind="stable")
        self.nodes = int(edge_index.max().item()) + 1
        self.rowptr = np.zeros(self.nodes + 1, dtype=np.int64)
        np.add.at(self.rowptr, source + 1, 1)
        self.rowptr = np.cumsum(self.rowptr, dtype=np.int64)
        self.col = target[order]
        self.weight = weight[order]
        self.rng = np.random.default_rng(seed)

    def sample(self, centers: np.ndarray, hops: int, fanout: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        selected = list(map(int, centers.tolist()))
        seen = set(selected)
        frontier = selected
        for _ in range(hops):
            next_frontier = []
            for node in frontier:
                start, end = self.rowptr[node], self.rowptr[node + 1]
                neighbors = self.col[start:end]
                if neighbors.size > fanout:
                    chosen = self.rng.choice(neighbors.size, size=fanout, replace=False)
                    neighbors = neighbors[chosen]
                for neighbor in neighbors.tolist():
                    if int(neighbor) not in seen:
                        seen.add(int(neighbor))
                        next_frontier.append(int(neighbor))
            selected.extend(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        node_ids = np.asarray(selected, dtype=np.int64)
        local = {int(node): index for index, node in enumerate(selected)}
        rows, cols, weights = [], [], []
        for source_node in selected:
            start, end = self.rowptr[source_node], self.rowptr[source_node + 1]
            for target_node, edge_value in zip(self.col[start:end], self.weight[start:end]):
                target_int = int(target_node)
                if target_int in local:
                    rows.append(local[source_node])
                    cols.append(local[target_int])
                    weights.append(float(edge_value))
        sub_edge_index = np.asarray([rows, cols], dtype=np.int64)
        sub_edge_weight = np.asarray(weights, dtype=np.float32)
        return node_ids, sub_edge_index, sub_edge_weight


def encode_centers_minibatch(model: GraphModel, x: Tensor, edge_index: Tensor, edge_weight: Tensor, centers: np.ndarray, args: argparse.Namespace, device: torch.device, sampler: CSRNeighborSampler) -> tuple[Tensor, np.ndarray]:
    """Encode centers through sampled subgraphs without retaining autograd graphs."""
    model.eval()
    node_embeddings, pooled_embeddings = [], []
    with torch.no_grad():
        for start in range(0, centers.size, args.batch_size):
            batch_centers = centers[start : start + args.batch_size]
            node_ids, sub_edges, sub_weights = sampler.sample(batch_centers, args.radius + 2, args.fanout)
            sub_x = x[torch.as_tensor(node_ids, device=device)]
            sub_edge_index = torch.as_tensor(sub_edges, device=device, dtype=torch.long)
            sub_edge_weight = torch.as_tensor(sub_weights, device=device)
            sub_z = model(sub_x, sub_edge_index, sub_edge_weight)
            center_positions = np.arange(batch_centers.size, dtype=np.int64)
            sub_pooled = local_embeddings(sub_z, sub_edge_index, center_positions, args.radius)
            node_embeddings.append(sub_z[: batch_centers.size].cpu())
            pooled_embeddings.append(sub_pooled.cpu())
    return torch.cat(node_embeddings), torch.cat(pooled_embeddings).numpy().astype(np.float32)


def train_supervised_encoder(model: GraphModel, x: Tensor, edge_index: Tensor, edge_weight: Tensor, centers: np.ndarray, labels: np.ndarray, train_idx: np.ndarray, args: argparse.Namespace, device: torch.device) -> tuple[nn.Module, dict[str, object]]:
    """Jointly optimize GNN message passing and the nonlinear label head."""
    sampler = CSRNeighborSampler(edge_index, edge_weight, args.seed)
    feature_dim = args.embedding_dim + 2
    classifier = MultiLabelClassifier(feature_dim, hidden_dim=args.classifier_hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.supervised_lr,
        weight_decay=1e-5,
    )
    train_labels = labels[train_idx]
    positive = torch.from_numpy(train_labels.sum(axis=0)).to(device).clamp_min(1.0)
    negative = torch.tensor(float(train_labels.shape[0]), device=device) - positive
    pos_weight = (negative.clamp_min(1.0) / positive).float()
    y = torch.from_numpy(train_labels).to(device)
    model.train()
    classifier.train()
    batch_size = args.batch_size
    train_rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.classifier_epochs + 1):
        epoch_loss = 0.0
        shuffled = train_rng.permutation(train_idx)
        for start in range(0, shuffled.size, batch_size):
            batch_positions = shuffled[start : start + batch_size]
            batch_centers = centers[batch_positions]
            node_ids, sub_edges, sub_weights = sampler.sample(batch_centers, args.radius + 2, args.fanout)
            sub_x = x[torch.as_tensor(node_ids, device=device)]
            sub_edge_index = torch.as_tensor(sub_edges, device=device, dtype=torch.long)
            sub_edge_weight = torch.as_tensor(sub_weights, device=device)
            center_positions = np.arange(batch_centers.size, dtype=np.int64)
            optimizer.zero_grad(set_to_none=True)
            sub_z = model(sub_x, sub_edge_index, sub_edge_weight)
            pooled = local_embeddings(sub_z, sub_edge_index, center_positions, args.radius)
            # ``batch_positions`` indexes the full center/label arrays, while
            # ``y`` only contains the compact training subset. Index the source
            # labels directly to avoid mixing global and local indices.
            batch_labels = torch.from_numpy(labels[batch_positions]).to(device)
            loss = F.binary_cross_entropy_with_logits(classifier(pooled), batch_labels, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(classifier.parameters()), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu()) * batch_positions.size / shuffled.size
        if epoch == 1 or epoch % max(1, args.classifier_epochs // 10) == 0:
            print(f"supervised epoch {epoch:03d}/{args.classifier_epochs} loss={epoch_loss:.5f}")
    classifier.eval()
    with torch.no_grad():
        _, train_pooled = encode_centers_minibatch(model, x, edge_index, edge_weight, centers[train_idx], args, device, sampler)
        train_probability = torch.sigmoid(classifier(torch.from_numpy(train_pooled).to(device)))
        train_prediction = (train_probability >= 0.5).float()
        accuracy = float((train_prediction == y).float().mean().cpu())
    return classifier, {"train_binary_accuracy": accuracy, "positive_counts": train_labels.sum(axis=0).tolist(), "train_size": int(train_idx.size), "batch_size": int(batch_size), "fanout": int(args.fanout), "training_mode": "end_to_end_supervised_subgraph_minibatch"}


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
    parser.add_argument("--batch-size", type=int, default=32, help="Number of center embeddings per supervised training batch (default: 32).")
    parser.add_argument("--fanout", type=int, default=15, help="Maximum neighbors sampled per node per hop in supervised subgraph batches.")
    parser.add_argument("--classifier-hidden-dim", type=int, default=64)
    parser.add_argument("--supervised-lr", type=float, default=1e-3)
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
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.fanout <= 0:
        raise SystemExit("--fanout must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    arrays, x, edge_index, edge_weight, input_path = load_graph(args.input, device)
    model = GraphModel(x.size(1), args.hidden_dim, args.embedding_dim).to(device)
    if args.task == "supervised":
        # Choose centers and labels before joint optimization. The labels are
        # fixed coordinates; the GNN and nonlinear head are then optimized
        # together against the train subset.
        with torch.no_grad():
            node_count = x.size(0)
            if args.centers == 0 or args.centers >= node_count:
                centers = np.arange(node_count, dtype=np.int64)
            else:
                centers = np.linspace(0, node_count - 1, args.centers, dtype=np.int64)
        labels = labels_for_centers(arrays, centers, args.labels)
        train_idx, test_idx = split_label_indices(labels, args.test_fraction, args.seed)
        classifier, classifier_stats = train_supervised_encoder(
            model, x, edge_index, edge_weight, centers, labels, train_idx, args, device
        )
        sampler = CSRNeighborSampler(edge_index, edge_weight, args.seed)
        z, pooled = encode_centers_minibatch(model, x, edge_index, edge_weight, centers, args, device, sampler)
    else:
        train_encoder(model, x, edge_index, edge_weight, args, device)
        z, centers, pooled = encode_and_pool(model, x, edge_index, edge_weight, args)
        labels = np.empty((0, 3), dtype=np.float32)
    stem = input_path.stem

    classifier_stats = {"task": args.task} if args.task == "unsupervised" else classifier_stats
    if args.task == "supervised":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(classifier.state_dict(), args.output_dir / f"{stem}_classifier.pt")
        test_pooled = pooled[test_idx]
        test_labels = labels[test_idx]
        classifier_stats["test"] = evaluate_classifier(classifier, test_pooled, test_labels, device)
        test_probabilities, test_novelty, test_candidate = novelty_scores(classifier, test_pooled, test_labels, device, args.novelty_threshold)
        classifier_stats["test"]["novel_candidate_count"] = int(test_candidate.sum())
        save_embeddings(args.output_dir, f"{stem}_random_test", arrays, centers[test_idx], z, test_pooled, test_labels, {"split": "random_test", "train_input": str(input_path), "test_fraction": args.test_fraction, "radius": args.radius, "embedding_dim": int(test_pooled.shape[1]), "novelty_threshold": args.novelty_threshold}, probabilities=test_probabilities, novelty_score=test_novelty, novel_candidate=test_candidate)
        classifier_stats["split"] = {"train_size": int(train_idx.size), "test_size": int(test_idx.size), "test_fraction": args.test_fraction}
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
