"""Joint OPCID/CHIN/CHID recognition and novel-region discovery.

Supports Cooler-labeled regions or paired rep1/rep2 NPZ datasets. CHIN and
CHID share the GNN/CNN fusion trunk, while their heads and supervised losses
remain label-specific. DataLoader workers prepare graph/contact patches on CPU.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def _json_default(value):
    """Convert NumPy/PyTorch scalar containers to JSON-compatible values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_evaluation_report(test_metrics, output_dir: Path, stem: str):
    """Write a compact three-class precision/recall/F1 evaluation report."""
    class_names = ("OPCID", "CHIN", "CHID")
    report = {
        "classes": list(class_names),
        "precision": [float(value) for value in test_metrics["precision"]],
        "recall": [float(value) for value in test_metrics["recall"]],
        "f1": [float(value) for value in test_metrics["f1"]],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}_evaluation_metrics.json"
    png_path = output_dir / f"{stem}_evaluation_metrics.png"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "--evaluate requires matplotlib to create the metrics visualization"
        ) from exc

    x = np.arange(len(class_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    bars = (
        ax.bar(x - width, report["precision"], width, label="Precision"),
        ax.bar(x, report["recall"], width, label="Recall"),
        ax.bar(x + width, report["f1"], width, label="F1"),
    )
    for group in bars:
        ax.bar_label(group, fmt="%.3f", padding=2, fontsize=8)
    ax.set_title("Test-set classification metrics")
    ax.set_ylabel("Score")
    ax.set_xticks(x, class_names)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return {"metrics_json": str(json_path), "metrics_png": str(png_path), **report}


try:
    from .GNN import GraphModel, MultiLabelClassifier, read_interval_labels
    from .region_GNN import (
        ContactCNN,
        CrossModalFusion,
        MultiScaleFusion,
        fused_region_embedding,
        _contact_channels,
        _diagonal_coordinate_view,
        _band_view,
        graph_context_bounds,
        multiscale_patches,
        region_graph,
    )
except ImportError:  # direct execution: python model/region_GNN_optimized.py
    from GNN import GraphModel, MultiLabelClassifier, read_interval_labels
    from region_GNN import (
        ContactCNN,
        CrossModalFusion,
        MultiScaleFusion,
        fused_region_embedding,
        _contact_channels,
        _diagonal_coordinate_view,
        _band_view,
        graph_context_bounds,
        multiscale_patches,
        region_graph,
    )


def build_positive_regions(labels_dir: Path, genome_end: int):
    """Return annotated regions with CHID encoded as the CHIN+CHID labels."""
    intervals = read_interval_labels(labels_dir)
    positives = []
    for _, start, end, label in intervals:
        left = max(0, int(start))
        right = min(int(genome_end), int(end))
        if right > left:
            positives.append((left, right, label))
    if not positives:
        raise ValueError("No positive regions found in the *_data.xlsx files")

    label_index = {"OPCID": 0, "CHIN": 1, "CHID": 2}
    samples = []
    for start, end, label in positives:
        target = np.zeros(3, dtype=np.float32)
        if label not in label_index:
            raise ValueError(f"Unsupported label {label!r}; expected OPCID, CHIN or CHID")
        target[label_index[label]] = 1.0
        # CHID denotes a cluster of CHIN regions. Encode this containment
        # relation explicitly instead of treating CHID and CHIN as disjoint.
        if label == "CHID":
            target[label_index["CHIN"]] = 1.0
        # Preserve overlapping annotations, e.g. a CHIN interval inside CHID.
        for other_start, other_end, other_label in positives:
            if other_label in label_index and start < other_end and end > other_start:
                target[label_index[other_label]] = 1.0
        samples.append((start, end, target))
    return samples


def _hard_negative_candidates(positives, genome_end):
    """Build real-coordinate hard-negative candidates around annotations."""
    occupied = [(start, end) for start, end, _ in positives]
    candidates = []

    def add(start, end):
        start, end = max(0, int(start)), min(int(genome_end), int(end))
        if end <= start or any(start < right and end > left for left, right in occupied):
            return
        if not any(start == left and end == right for left, right, _ in candidates):
            candidates.append((start, end, np.zeros(3, dtype=np.float32)))

    for start, end, label in positives:
        length = max(1, end - start)
        offset = max(1, length // 4)
        add(start - length, start)
        add(end, end + length)
        add(start - length - offset, start - offset)
        add(end + offset, end + length + offset)
        if label[0] > 0.5 or label[1] > 0.5:
            add(start - 2 * length, start - length)
            add(end + length, end + 2 * length)

    chin = [(start, end) for start, end, label in positives if label[1] > 0.5]
    for left_start, left_end in chin:
        for right_start, right_end in chin:
            if left_start >= right_start or right_start <= left_end:
                continue
            gap = right_start - left_end
            width = max(left_end - left_start, right_end - right_start)
            if gap <= 4 * max(1, width):
                center = (left_end + right_start) // 2
                add(center - width // 2, center + (width + 1) // 2)
    return candidates


def build_training_regions(
    labels_dir: Path,
    genome_end: int,
    seed: int,
    include_background_negatives: bool,
    include_hard_negative: bool = False,
):
    """Build a balanced set with one negative for every annotated structure.

    Hard negatives are selected first when requested; random non-overlapping
    windows fill the remainder so the returned training pool is always
    ``N_positive + N_negative`` with equal counts.
    """
    positives = build_positive_regions(labels_dir, genome_end)
    rng = np.random.default_rng(seed)
    occupied = [(start, end) for start, end, _ in positives]
    negatives = []
    target_negative_count = len(positives)
    if include_hard_negative:
        hard = _hard_negative_candidates(positives, genome_end)
        rng.shuffle(hard)
        negatives.extend(hard[:target_negative_count])
        occupied.extend((start, end) for start, end, _ in negatives)
    attempts = 0
    while len(negatives) < target_negative_count and attempts < max(1, len(positives) * 2000):
        attempts += 1
        positive_start, positive_end, _ = positives[int(rng.integers(len(positives)))]
        length = positive_end - positive_start
        if genome_end <= length:
            break
        start = int(rng.integers(0, genome_end - length + 1))
        end = start + length
        if any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied):
            continue
        negatives.append((start, end, np.zeros(3, dtype=np.float32)))
        occupied.append((start, end))
    if len(negatives) < target_negative_count:
        raise ValueError(
            f"Could only construct {len(negatives)} non-overlapping negatives for "
            f"{len(positives)} positives; reduce annotation span or genome window size."
        )
    return positives + negatives[:target_negative_count], target_negative_count


def region_contact_features(patches):
    """Extract compact region statistics useful for separating CHIN/CHID.

    The statistics are computed from the original square view at each of the
    250/500/1000 bp scales. They complement learned CNN tokens with intensity,
    diagonal-band and off-diagonal cluster information.
    """
    features = []
    for patch_views in patches:
        original = patch_views[0]  # [channels=3, H, W]
        log_contact = original[0]
        offset_z = original[1]
        diagonal_distance = original[2]
        diagonal = torch.diagonal(log_contact, dim1=-2, dim2=-1)
        near_band = (diagonal_distance <= 0.20).to(log_contact.dtype)
        far_band = (diagonal_distance >= 0.50).to(log_contact.dtype)
        features.extend(
            (
                log_contact.mean(),
                log_contact.std(unbiased=False),
                log_contact.amax(),
                offset_z.abs().mean(),
                (offset_z > 1.0).to(log_contact.dtype).mean(),
                diagonal.mean(),
                (log_contact * near_band).sum() / near_band.sum().clamp_min(1.0),
                (log_contact * far_band).sum() / far_band.sum().clamp_min(1.0),
            )
        )
    return torch.stack(features)


def cluster_region_features(patches):
    """Return compact features describing a cluster of CHIN-like bands."""
    features = []
    for patch_views in patches:
        # The diagonal-coordinate and band views expose separated CHIN-like
        # peaks more directly than the original square view.
        view = patch_views[1]
        signal = view[1]
        high = signal > 1.0
        row_mass = high.to(signal.dtype).mean(dim=-1)
        col_mass = high.to(signal.dtype).mean(dim=-2)
        features.extend((
            high.to(signal.dtype).mean(),
            row_mass.amax(),
            (row_mass > 0.10).to(signal.dtype).sum() / row_mass.numel(),
            col_mass.amax(),
            (col_mass > 0.10).to(signal.dtype).sum() / col_mass.numel(),
            (signal * high).sum() / high.sum().clamp_min(1.0),
        ))
    return torch.stack(features)


class BinaryHead(torch.nn.Module):
    """Small class-specific sigmoid-logit head."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class OPCIDVerifierCNN(torch.nn.Module):
    """Second-stage OPCID-only verifier operating on the square contact view."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.BatchNorm2d(16), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, hidden_dim, 3, padding=1), torch.nn.BatchNorm2d(hidden_dim), torch.nn.ReLU(),
        )
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Dropout(0.35),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, square_patch):
        return self.head(self.features(square_patch)).squeeze(-1)


class SpecializedRegionModel(torch.nn.Module):
    """Class-specific architecture for square, diagonal, and cluster signals."""

    def __init__(self, node_dim, hidden_dim, embedding_dim, classifier_hidden_dim, enable_opcid_verifier=False):
        super().__init__()
        self.gnn = GraphModel(node_dim, hidden_dim, embedding_dim)
        self.opcid_cnn = torch.nn.ModuleList([ContactCNN(embedding_dim) for _ in (250, 500, 1000)])
        self.chin_cnn = torch.nn.ModuleList([ContactCNN(embedding_dim) for _ in (250, 500, 1000)])
        self.cross_attention = CrossModalFusion(embedding_dim, 64, embedding_dim)
        self.embedding_dim = embedding_dim
        self.opcid_head = BinaryHead(embedding_dim, classifier_hidden_dim)
        self.opcid_verifier = OPCIDVerifierCNN() if enable_opcid_verifier else None
        # CrossModalFusion returns 2*embedding_dim; append graph topology and
        # diagonal statistics for CHIN, then cluster statistics for CHID.
        self.chin_head = BinaryHead(2 * embedding_dim + 2 + 24, classifier_hidden_dim)
        self.chid_head = BinaryHead(2 * embedding_dim + 2 + 24 + 18, classifier_hidden_dim)

    def verifier_logit(self, patches):
        if self.opcid_verifier is None:
            return None
        logits = []
        for patch_views in patches:
            square_contact = patch_views[0:1, 0:1]
            logits.append(self.opcid_verifier(square_contact).squeeze(0))
        return torch.stack(logits).mean()

    def verifier_logits(self, patches, augment=False):
        if self.opcid_verifier is None:
            return None
        logits = []
        for patch_views in patches:
            square_contact = patch_views[0:1, 0:1]
            if augment:
                if torch.rand((), device=square_contact.device) > 0.5:
                    square_contact = square_contact.flip(-1)
                if torch.rand((), device=square_contact.device) > 0.5:
                    square_contact = square_contact.flip(-2)
                if torch.rand((), device=square_contact.device) > 0.5:
                    square_contact = square_contact + torch.randn_like(square_contact) * 0.02
                if torch.rand((), device=square_contact.device) > 0.5:
                    square_contact = square_contact * torch.empty((), device=square_contact.device).uniform_(0.85, 1.15)
            logits.append(self.opcid_verifier(square_contact).squeeze(0))
        return torch.stack(logits).mean()

    @staticmethod
    def _augment_opcid_patch(patch):
        # Match OPCID 02_train_cnn.py: flip/noise/contrast are applied to the
        # contact image only; derived channels are then rebuilt consistently.
        contact = patch[:, 0:1]
        if torch.rand((), device=patch.device) > 0.5:
            contact = contact.flip(-1)
        if torch.rand((), device=patch.device) > 0.5:
            contact = contact.flip(-2)
        if torch.rand((), device=patch.device) > 0.5:
            contact = contact + torch.randn_like(contact) * 0.02
        if torch.rand((), device=patch.device) > 0.5:
            contact = contact * torch.empty((), device=patch.device).uniform_(0.85, 1.15)
        batch, _, height, width = contact.shape
        standardized = torch.zeros_like(contact)
        for offset in range(width):
            upper = torch.diagonal(contact[:, 0], offset=offset, dim1=-2, dim2=-1)
            upper_norm = (upper - upper.mean(dim=-1, keepdim=True)) / upper.std(
                dim=-1, unbiased=False, keepdim=True
            ).clamp_min(1e-6)
            for sample in range(batch):
                standardized[sample, 0].diagonal(offset=offset).copy_(upper_norm[sample])
            if offset:
                lower = torch.diagonal(contact[:, 0], offset=-offset, dim1=-2, dim2=-1)
                lower_norm = (lower - lower.mean(dim=-1, keepdim=True)) / lower.std(
                    dim=-1, unbiased=False, keepdim=True
                ).clamp_min(1e-6)
                for sample in range(batch):
                    standardized[sample, 0].diagonal(offset=-offset).copy_(lower_norm[sample])
        rows = torch.arange(height, device=patch.device)[:, None]
        cols = torch.arange(width, device=patch.device)[None, :]
        distance = (rows - cols).abs().float() / max(height - 1, 1)
        distance = distance.unsqueeze(0).unsqueeze(0).expand(batch, -1, -1, -1)
        return torch.cat((contact, standardized, distance), dim=1)

    def encode(self, graph, patches, augment_opcid=False):
        z = self.gnn(graph[0], graph[1], graph[2])
        topology = torch.log1p(torch.tensor([graph[0].size(0), graph[1].size(1)], dtype=z.dtype, device=z.device))
        # Keep a batch dimension for Conv2d/BatchNorm2d.  Each scale contains
        # [views=3, channels=3, H, W], so the square view is [1, 3, H, W].
        opcid_embeddings = []
        for branch, patch_views in zip(self.opcid_cnn, patches):
            square = patch_views[0:1]
            if augment_opcid:
                square = self._augment_opcid_patch(square)
            opcid_embeddings.append(branch(square).squeeze(0))
        opcid = torch.stack(opcid_embeddings).mean(dim=0)
        tokens = []
        for branch, patch_views in zip(self.chin_cnn, patches):
            geometric = patch_views[1:3]
            view_tokens = branch.tokens(geometric)
            tokens.append(view_tokens.reshape(1, -1, view_tokens.size(-1)))
        chin_cross = self.cross_attention(z, torch.cat(tokens, dim=1).squeeze(0))
        contact = region_contact_features(patches).to(z.device)
        cluster = cluster_region_features(patches).to(z.device)
        chin = torch.cat((chin_cross, topology, contact))
        chid = torch.cat((chin_cross, topology, contact, cluster))
        return opcid, chin, chid

    def forward(self, graph, patches):
        opcid, chin, chid = self.encode(graph, patches)
        return torch.stack((self.opcid_head(opcid), self.chin_head(chin), self.chid_head(chid)))


def evaluate_with_thresholds(logits, labels, thresholds, verifier_logits=None, verifier_threshold=None):
    """Evaluate each class with its own decision threshold."""
    probability = torch.sigmoid(torch.stack(logits)).detach().cpu().numpy()
    labels = np.asarray(labels, dtype=np.float32)
    threshold_array = np.asarray(thresholds, dtype=np.float32).reshape(1, 3)
    prediction = (probability >= threshold_array).astype(np.float32)
    verifier_probability = None
    if verifier_logits is not None and verifier_threshold is not None:
        verifier_probability = torch.sigmoid(torch.stack(verifier_logits)).detach().cpu().numpy()
        prediction[:, 0] = prediction[:, 0] * (verifier_probability >= float(verifier_threshold)).astype(np.float32)
    tp = (prediction * labels).sum(axis=0)
    precision = tp / np.maximum(prediction.sum(axis=0), 1.0)
    recall = tp / np.maximum(labels.sum(axis=0), 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    confusion = []
    for class_index in range(labels.shape[1]):
        truth = labels[:, class_index] >= 0.5
        predicted = prediction[:, class_index] >= 0.5
        confusion.append([
            [int(np.logical_and(~truth, ~predicted).sum()), int(np.logical_and(~truth, predicted).sum())],
            [int(np.logical_and(truth, ~predicted).sum()), int(np.logical_and(truth, predicted).sum())],
        ])
    result = {
        "binary_accuracy": float((prediction == labels).mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "confusion_matrix_tn_fp_fn_tp": confusion,
        "thresholds": threshold_array.reshape(-1).tolist(),
    }
    if verifier_probability is not None:
        result["opcid_verifier_threshold"] = float(verifier_threshold)
        result["opcid_verifier_positive_predictions"] = int((verifier_probability >= float(verifier_threshold)).sum())
    return result


def _load_npz_contact_bundle(paths):
    """Merge OPCID/CHIN/CHID single-label NPZ files by coordinates."""
    records = {}
    for class_index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as archive:
            images = np.asarray(archive["images"], dtype=np.float32)
            labels = np.asarray(archive["labels"], dtype=np.int64)
            coords = np.asarray(archive["coords"], dtype=np.int64)
        if not (len(images) == len(labels) == len(coords)):
            raise ValueError(f"Mismatched lengths in contact dataset: {path}")
        for image, label, coord in zip(images, labels, coords):
            key = (int(coord[0]), int(coord[1]), int(coord[2]))
            record = records.setdefault(key, {"image": image, "label": np.zeros(3, dtype=np.float32)})
            record["label"][class_index] = float(label)
    regions, images = [], []
    for (start, end, _center), record in sorted(records.items()):
        label = record["label"]
        if label[2] > 0:
            label[1] = 1.0
        regions.append((start, end, label))
        images.append(record["image"])
    if not regions:
        raise ValueError("No samples found in contact NPZ datasets")
    return regions, np.stack(images).astype(np.float32, copy=False)


def _split_npz_validation(regions, images, fraction, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(regions))
    n_val = min(max(1, int(round(len(indices) * fraction))), len(indices) - 1)
    val, fit = indices[:n_val], indices[n_val:]
    return ([regions[i] for i in fit], images[fit], [regions[i] for i in val], images[val])


def add_npz_hard_negative_repeats(regions, images, seed, repeat_count=None):
    """Repeat real zero-label rep1 windows with the strongest structure proxies."""
    labels = np.asarray([region[2] for region in regions], dtype=np.float32)
    candidates = np.flatnonzero(labels.sum(axis=1) == 0)
    if candidates.size == 0:
        return regions, images, 0
    positives = np.flatnonzero(labels.sum(axis=1) > 0)
    centers = np.asarray(
        [(regions[i][0] + regions[i][1]) * 0.5 for i in positives], dtype=np.float32
    )
    scores = []
    for index in candidates:
        image = np.nan_to_num(np.asarray(images[index], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        diagonal = np.diagonal(image, axis1=-2, axis2=-1)
        off_diagonal = image.copy()
        diagonal_indices = np.arange(min(image.shape[-2:]))
        off_diagonal[..., diagonal_indices, diagonal_indices] = 0.0
        center = (regions[index][0] + regions[index][1]) * 0.5
        width = max(1.0, float(regions[index][1] - regions[index][0]))
        proximity = 0.0 if centers.size == 0 else 1.0 / (1.0 + float(np.min(np.abs(centers - center))) / width)
        row_peak = np.max(image, axis=-1).mean()
        col_peak = np.max(image, axis=-2).mean()
        scores.append(
            float(image.mean()) + 0.5 * float(image.std()) + 0.25 * float(image.max())
            + 0.25 * float(diagonal.mean()) + 0.25 * float(off_diagonal.mean())
            + 0.25 * float(max(row_peak, col_peak)) + 0.5 * proximity
        )
    rng = np.random.default_rng(seed)
    order = np.argsort(-(np.asarray(scores) + rng.random(len(scores)) * 1e-9))
    if repeat_count is None:
        repeat_count = max(1, int(labels.any(axis=1).sum()))
    selected = candidates[order[:min(int(repeat_count), len(order))]]
    return (
        list(regions) + [regions[int(i)] for i in selected],
        np.concatenate((images, images[selected]), axis=0),
        len(selected),
    )


def balance_npz_training_regions(regions, images, seed, include_hard_negative=False):
    """Balance the current-dataset rep1 pool at one negative per positive.

    The existing NPZ files contain 344 labeled structures and 287 explicit
    zero-label windows. Missing negatives are filled by repeating the most
    structure-like zero-label windows when hard-negative mode is enabled, or
    by deterministic random oversampling otherwise. No labels or images are
    synthesized for positive structures.
    """
    labels = np.asarray([region[2] for region in regions], dtype=np.float32)
    positive_indices = np.flatnonzero(labels.sum(axis=1) > 0)
    negative_indices = np.flatnonzero(labels.sum(axis=1) == 0)
    if positive_indices.size == 0:
        raise ValueError("Current dataset contains no positive regions")
    target = int(positive_indices.size)
    rng = np.random.default_rng(seed)
    if negative_indices.size == 0:
        raise ValueError("Current dataset contains no zero-label negative regions")
    if negative_indices.size >= target:
        chosen = rng.choice(negative_indices, size=target, replace=False)
    else:
        if include_hard_negative:
            candidate_regions, candidate_images, _ = add_npz_hard_negative_repeats(
                [regions[int(i)] for i in negative_indices], images[negative_indices],
                seed, repeat_count=target - int(negative_indices.size),
            )
            # Keep every available negative once and append selected hard repeats.
            balanced_regions = [regions[int(i)] for i in positive_indices] + candidate_regions
            balanced_images = np.concatenate((images[positive_indices], candidate_images), axis=0)
            order = rng.permutation(len(balanced_regions))
            return [balanced_regions[int(i)] for i in order], balanced_images[order], int(target - negative_indices.size)
        chosen = rng.choice(negative_indices, size=target, replace=True)
    balanced_regions = [regions[int(i)] for i in positive_indices] + [regions[int(i)] for i in chosen]
    balanced_images = np.concatenate((images[positive_indices], images[chosen]), axis=0)
    order = rng.permutation(len(balanced_regions))
    return [balanced_regions[int(i)] for i in order], balanced_images[order], max(0, target - int(negative_indices.size))


def effective_number_weights(labels, beta=0.999):
    counts = np.asarray(labels, dtype=np.float32).sum(axis=0)
    weights = np.zeros(3, dtype=np.float32)
    present = counts > 0
    weights[present] = (1.0 - beta) / (1.0 - np.power(beta, counts[present]))
    if present.any():
        weights[present] *= present.sum() / weights[present].sum()
    return weights


def asymmetric_loss_per_class(
    logits,
    targets,
    class_weights,
    positive_weights=None,
    gamma_pos=0.0,
    gamma_neg=2.0,
    clip=0.05,
):
    """Compute ASL with optional per-label positive-example compensation.

    ``positive_weights`` is the ASL equivalent of BCEWithLogitsLoss's
    ``pos_weight``: it increases the cost of missing a positive label without
    increasing the contribution of easy negative examples.
    """
    targets = targets.float()
    probabilities = torch.sigmoid(logits).clamp(min=1e-8, max=1.0)
    negative = (1.0 - probabilities + clip).clamp(max=1.0) if clip > 0 else 1.0 - probabilities
    if positive_weights is None:
        positive_weights = torch.ones(logits.size(-1), dtype=logits.dtype, device=logits.device)
    positive_term = (
        targets
        * (1.0 - probabilities).pow(gamma_pos)
        * torch.log(probabilities)
        * positive_weights.view(1, -1)
    )
    negative_term = (
        (1.0 - targets)
        * probabilities.pow(gamma_neg)
        * torch.log(negative.clamp_min(1e-8))
    )
    loss = -(positive_term + negative_term)
    return (loss * class_weights.view(1, -1)).mean(dim=0)


def supervised_contrastive_loss(embedding, labels, temperature=0.1):
    if embedding.size(0) < 2:
        return embedding.sum() * 0.0
    normalized = F.normalize(embedding, dim=1)
    similarity = normalized @ normalized.T / temperature
    eye = torch.eye(embedding.size(0), dtype=torch.bool, device=embedding.device)
    positive = (labels @ labels.T > 0) & ~eye
    logits = similarity.masked_fill(eye, -1e9)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    count = positive.sum(dim=1)
    valid = count > 0
    if not valid.any():
        return embedding.sum() * 0.0
    return -(log_probability * positive.float()).sum(dim=1)[valid].div(count[valid]).mean()


def center_loss(embedding, labels):
    losses = []
    for class_index in range(labels.size(1)):
        mask = labels[:, class_index] > 0.5
        if mask.any():
            values = embedding[mask]
            center = values.mean(dim=0).detach()
            losses.append((values - center).pow(2).mean())
    return torch.stack(losses).mean() if losses else embedding.sum() * 0.0


def fit_embedding_prototypes(embeddings, labels):
    matrix = torch.stack(embeddings).numpy().astype(np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    prototypes = {}
    for index, name in enumerate(("OPCID", "CHIN", "CHID")):
        mask = labels[:, index] > 0.5
        if mask.any():
            prototypes[name] = (matrix[mask] / scale).mean(axis=0).astype(np.float32)
    return {"scale": scale, "prototypes": prototypes}


def embedding_novelty_distance(embeddings, prototype_state):
    if not embeddings or not prototype_state["prototypes"]:
        return np.zeros(0, dtype=np.float32)
    matrix = torch.stack(embeddings).numpy().astype(np.float32) / prototype_state["scale"]
    prototype_matrix = np.stack(list(prototype_state["prototypes"].values()))
    return np.min(((matrix[:, None, :] - prototype_matrix[None, :, :]) ** 2).sum(axis=2) ** 0.5, axis=1)


def _select_threshold_by_precision(logits, labels, minimum_recall=0.0, fallback=0.5):
    """Maximize precision among thresholds meeting the recall floor."""
    probabilities = torch.sigmoid(torch.stack(logits)).detach().cpu().numpy()[:, 0]
    labels = np.asarray(labels, dtype=np.float32)
    best = None
    for threshold in np.arange(0.05, 0.951, 0.01):
        prediction = probabilities >= threshold
        tp = float(np.logical_and(prediction, labels > 0).sum())
        precision = tp / max(float(prediction.sum()), 1.0)
        recall = tp / max(float((labels > 0).sum()), 1.0)
        if recall < minimum_recall:
            continue
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        candidate = (precision, recall, float(threshold), f1)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        positive_count = int((labels > 0).sum())
        return fallback, None, None, None, {
            "feasible": False,
            "reason": "no threshold met the requested recall floor",
            "validation_positive_count": positive_count,
        }
    precision, recall, threshold, f1 = best
    return threshold, precision, recall, f1, {
        "feasible": True,
        "validation_positive_count": int((labels > 0).sum()),
    }


def _select_opcid_cascade_thresholds(main_logits, verifier_logits, labels, minimum_recall, fallbacks):
    """Tune both OPCID stages jointly so their combined recall meets the floor."""
    main_probability = torch.sigmoid(torch.stack(main_logits)).detach().cpu().numpy()[:, 0]
    verifier_probability = torch.sigmoid(torch.stack(verifier_logits)).detach().cpu().numpy()
    positive = np.asarray(labels, dtype=np.float32) > 0.5
    candidates = np.arange(0.05, 0.951, 0.02)
    best = None
    for main_threshold in candidates:
        main_prediction = main_probability >= main_threshold
        for verifier_threshold in candidates:
            prediction = main_prediction & (verifier_probability >= verifier_threshold)
            tp = float(np.logical_and(prediction, positive).sum())
            recall = tp / max(float(positive.sum()), 1.0)
            if recall < minimum_recall:
                continue
            precision = tp / max(float(prediction.sum()), 1.0)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            candidate = (precision, recall, float(main_threshold), float(verifier_threshold), f1)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return fallbacks[0], fallbacks[1], None, None, None, {
            "feasible": False,
            "reason": "no threshold pair met the requested cascade recall floor",
            "validation_positive_count": int(positive.sum()),
        }
    precision, recall, main_threshold, verifier_threshold, f1 = best
    return main_threshold, verifier_threshold, precision, recall, f1, {
        "feasible": True,
        "validation_positive_count": int(positive.sum()),
    }


class RegionDataset(Dataset):
    """Prepare local graphs and either Cooler or precomputed NPZ patches."""

    def __init__(self, regions, shared_arrays, cool_path: Path | None, patch_size: int, contact_images=None):
        self.regions = regions
        # CPU tensors are shared once when workers are enabled. This avoids
        # serializing a full graph copy into every Windows worker process.
        self.shared_arrays = shared_arrays
        self.cool_path = str(cool_path)
        self.patch_size = patch_size
        self.contact_images = contact_images
        self._pid = None
        self._arrays = None
        self._cool_matrix = None
        self._cooler = None

    def __len__(self):
        return len(self.regions)

    def _init_process_resources(self):
        pid = os.getpid()
        if self._pid == pid and self._arrays is not None and (
            self.contact_images is not None or self._cool_matrix is not None
        ):
            return

        # Do not open an HDF5/Cooler handle in the parent and pass it to child
        # processes. Every worker opens its own handle after it starts.
        self._arrays = {key: tensor.numpy() for key, tensor in self.shared_arrays.items()}
        if self.contact_images is None:
            import cooler
            self._cooler = cooler.Cooler(self.cool_path)
            self._cool_matrix = self._cooler.matrix(balance=False, sparse=False)
        self._pid = pid

    def __getitem__(self, index):
        self._init_process_resources()
        start, end, label = self.regions[index]
        arrays = self._arrays
        edge_index = arrays["edge_index"]
        edge_weight = arrays["edge_weight"]
        graph_start, graph_end = graph_context_bounds(start, end, arrays, 500)
        graph = region_graph(
            arrays,
            edge_index,
            edge_weight,
            graph_start,
            graph_end,
            torch.device("cpu"),
        )
        if graph is None:
            return None

        if self.contact_images is not None:
            base = torch.from_numpy(np.asarray(self.contact_images[index], dtype=np.float32))
            raw = torch.expm1(base).clamp_min(0)
            channels = _contact_channels(raw)
            views = (channels, _diagonal_coordinate_view(channels), _band_view(channels))
            one_scale = torch.stack(tuple(
                F.interpolate(view.unsqueeze(0), size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False).squeeze(0)
                for view in views
            ))
            patches = tuple(one_scale.clone() for _ in (250, 500, 1000))
        else:
            patches = multiscale_patches(
                self._cool_matrix, arrays, (start, end), (250, 500, 1000),
                self.patch_size, torch.device("cpu")
            )
        if patches is None:
            return None

        return {
            "x": graph[0],
            "edge_index": graph[1],
            "edge_weight": graph[2],
            "patches": tuple(patches),
            "label": torch.from_numpy(np.asarray(label, dtype=np.float32)),
            "start": int(start),
            "end": int(end),
            "index": int(index),
        }


def _collate_region_batch(samples):
    """Keep variable-size graphs as a list; discard unusable regions."""
    return [sample for sample in samples if sample is not None]


def _worker_init_fn(_worker_id):
    # Prevent every worker from creating its own large OpenMP thread pool.
    torch.set_num_threads(1)


def _make_loader(
    dataset,
    *,
    batch_size,
    shuffle,
    num_workers,
    pin_memory,
    persistent_workers,
    prefetch_factor,
    generator=None,
):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers and num_workers > 0,
        "collate_fn": _collate_region_batch,
        "worker_init_fn": _worker_init_fn if num_workers > 0 else None,
        "generator": generator,
    }
    # PyTorch only accepts prefetch_factor when worker processes are enabled.
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def _move_sample_to_device(sample, device, non_blocking):
    graph = (
        sample["x"].to(device, non_blocking=non_blocking),
        sample["edge_index"].to(device, non_blocking=non_blocking),
        sample["edge_weight"].to(device, non_blocking=non_blocking),
    )
    patches = [patch.to(device, non_blocking=non_blocking) for patch in sample["patches"]]
    label = sample["label"].to(device, non_blocking=non_blocking)
    return graph, patches, label


def load_graph_archive(path, num_workers=0):
    tensors = {}
    with np.load(path, allow_pickle=False) as archive:
        for key in ("x", "node_start", "node_end", "edge_index", "edge_weight"):
            if key not in archive:
                raise ValueError(f"Graph archive is missing required array: {key}")
            tensor = torch.from_numpy(np.ascontiguousarray(archive[key]))
            if num_workers > 0:
                try:
                    tensor.share_memory_()
                except RuntimeError as exc:
                    raise SystemExit("Could not place graph arrays in shared memory; try --num-workers 0") from exc
            tensors[key] = tensor
    return tensors


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--cooler", action="store_true", help="Use Cooler + Excel annotations and the custom coordinate split.")
    mode_group.add_argument("--current-dataset", action="store_true", help="Use the existing rep1/rep2 class NPZ datasets.")
    parser.add_argument("--input", type=Path, default=None, help="Rep1 graph archive.")
    parser.add_argument("--test-input", type=Path, default=None, help="Rep2 graph archive in current-dataset mode.")
    parser.add_argument("--cool-input", type=Path, default=None, help="Cooler file used for CNN matrix patches in --cooler mode.")
    parser.add_argument("--opcid-train-npz", type=Path, default=None)
    parser.add_argument("--chin-train-npz", type=Path, default=None)
    parser.add_argument("--chid-train-npz", type=Path, default=None)
    parser.add_argument("--opcid-test-npz", type=Path, default=None)
    parser.add_argument("--chin-test-npz", type=Path, default=None)
    parser.add_argument("--chid-test-npz", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=root / "data/datasets")
    parser.add_argument("--output-dir", type=Path, default=root / "data/region_embeddings")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Write only test precision/recall/F1 metrics and a visualization after evaluation.",
    )
    parser.add_argument(
        "--evaluate-output-dir",
        type=Path,
        default=None,
        help="Directory for the --evaluate JSON and PNG report; defaults to <output-dir>/evaluation.",
    )
    parser.add_argument("--context", type=int, default=500, help="Recorded legacy value; GNN context is fixed at 500 bp and CNN contexts are 250/500/1000 bp.")
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--dev-earlystop", action="store_true", help="Use rep1 85/15 validation early stopping and rep2 as test.")
    parser.add_argument("--earlystop-patience", type=int, default=8)
    parser.add_argument("--threshold-finetune", action="store_true", help="Tune thresholds on the mode-specific validation split: maximize precision subject to the recall floors.")
    parser.add_argument("--opcid-min-recall", type=float, default=0.50, help="Minimum OPCID validation recall required during threshold tuning.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--cnn-dim", type=int, default=128, help="Per-scale CNN token projection width (kept for compatibility).")
    parser.add_argument("--patch-size", type=int, default=32, help="Fixed CNN patch size; 32 reduces overfitting and attention cost for small datasets.")
    parser.add_argument("--classifier-hidden-dim", type=int, default=32)
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asl-gamma-neg", type=float, default=2.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--class-balanced-beta", type=float, default=0.999)
    parser.add_argument(
        "--opcid-pos-weight",
        type=float,
        default=-1.0,
        help="OPCID positive-loss weight; <=0 automatically uses OPCID negatives/positives like old.py.",
    )
    parser.add_argument(
        "--max-opcid-pos-weight",
        type=float,
        default=20.0,
        help="Safety cap for the automatically computed OPCID positive-loss weight.",
    )
    parser.add_argument("--hier-loss-weight", type=float, default=0.2)
    parser.add_argument("--supcon-loss-weight", type=float, default=0.1)
    parser.add_argument("--center-loss-weight", type=float, default=0.005)
    parser.add_argument("--supcon-temperature", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4, help="CPU worker processes used to prepare regions; set 0 to disable multiprocessing.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Batches prefetched per worker.")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=True)
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--threshold", type=float, default=0.5, help="Legacy global fallback threshold; class-specific defaults below take precedence.")
    parser.add_argument("--opcid-threshold", type=float, default=0.55, help="OPCID threshold; raised slightly because current OPCID precision is low.")
    parser.add_argument("--opcid-verifier", action="store_true", help="Train and apply a second-stage square-contact CNN verifier for OPCID.")
    parser.add_argument("--opcid-verifier-threshold", type=float, default=0.60, help="Second-stage OPCID verifier threshold.")
    parser.add_argument("--chin-threshold", type=float, default=0.50, help="CHIN threshold.")
    parser.add_argument("--chid-threshold", type=float, default=0.40)
    parser.add_argument("--novelty-threshold", type=float, default=0.55)
    parser.add_argument(
        "--include-background-negatives",
        action="store_true",
        help="Compatibility flag; balanced training always includes one negative per positive, with hard negatives preferred when requested.",
    )
    parser.add_argument("--include-hard-negative", action="store_true")
    parser.add_argument("--chin-min-recall", type=float, default=0.85)
    parser.add_argument("--chid-min-recall", type=float, default=0.70)
    parser.add_argument("--novelty-distance-threshold", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scan", action="store_true", help="After training, scan the whole chromosome.")
    parser.add_argument("--scan-window", type=int, default=0, help="Window size in bp; 0 uses the median positive region size.")
    parser.add_argument("--scan-step", type=int, default=500)
    args = parser.parse_args()
    if args.current_dataset:
        args.input = args.input or root / "data/graphs/GSE272159_37C_rep1.mapq_30.10_top40.npz"
        args.test_input = args.test_input or root / "data/graphs/GSE272159_37C_rep2.mapq_30.10_top40.npz"
        args.opcid_train_npz = args.opcid_train_npz or root / "model/opcid/data_out/rep1_data.npz"
        args.opcid_test_npz = args.opcid_test_npz or root / "model/opcid/data_out/rep2_data.npz"
        args.chin_train_npz = args.chin_train_npz or root / "data/data_out/chin/chin_rep1_data.npz"
        args.chin_test_npz = args.chin_test_npz or root / "data/data_out/chin/chin_rep2_data.npz"
        args.chid_train_npz = args.chid_train_npz or root / "data/data_out/chid/chid_rep1_data.npz"
        args.chid_test_npz = args.chid_test_npz or root / "data/data_out/chid/chid_rep2_data.npz"
    npz_paths = (
        args.opcid_train_npz, args.chin_train_npz, args.chid_train_npz,
        args.opcid_test_npz, args.chin_test_npz, args.chid_test_npz,
    )
    npz_mode = bool(args.current_dataset)
    if args.cooler:
        if args.input is None or args.cool_input is None:
            raise SystemExit("--cooler requires --input and --cool-input")
        if any(path is not None for path in npz_paths):
            raise SystemExit("--cooler cannot be combined with current-dataset NPZ arguments")
    if args.current_dataset and args.cool_input is not None:
        raise SystemExit("--current-dataset uses only the prepared NPZ datasets; do not pass --cool-input")
    if npz_mode:
        if any(path is None for path in npz_paths) or args.test_input is None:
            raise SystemExit("--current-dataset requires all six class NPZ files and --test-input")
        missing = [str(path) for path in npz_paths if not path.is_file()]
        missing += [str(path) for path in (args.input, args.test_input) if not path.is_file()]
        if missing:
            raise SystemExit("Missing NPZ-mode input files: " + ", ".join(missing))
    if args.dev_earlystop or args.threshold_finetune:
        if not 0 < args.validation_fraction < 1:
            raise SystemExit("validation-fraction must be between 0 and 1")
    if args.scan and args.cool_input is None:
        raise SystemExit("--scan requires --cool-input to scan unseen chromosome windows")
    if args.earlystop_patience <= 0:
        raise SystemExit("earlystop-patience must be positive")
    if not all(0 <= value <= 1 for value in (args.opcid_min_recall, args.chin_min_recall, args.chid_min_recall)):
        raise SystemExit("minimum recall values must be between 0 and 1")
    if args.context < 0 or args.batch_size <= 0 or args.epochs <= 0:
        raise SystemExit("context, batch-size and epochs must be valid positive values")
    if args.num_workers < 0 or args.prefetch_factor <= 0:
        raise SystemExit("num-workers must be >= 0 and prefetch-factor must be > 0")
    if args.embedding_dim <= 0 or args.embedding_dim % 4 != 0:
        raise SystemExit("embedding-dim must be a positive multiple of 4 because cross-attention uses 4 heads")
    if args.max_opcid_pos_weight <= 0:
        raise SystemExit("max-opcid-pos-weight must be positive")
    if args.opcid_pos_weight > 0 and not np.isfinite(args.opcid_pos_weight):
        raise SystemExit("opcid-pos-weight must be finite")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but this PyTorch build/runtime has no available CUDA device.")

    cooler = None
    if not npz_mode or args.scan:
        try:
            import cooler as cooler_module
            cooler = cooler_module
        except ImportError as exc:
            raise SystemExit("CNN matrix input requires cooler; install it in the active environment.") from exc

    array_tensors = load_graph_archive(args.input, args.num_workers)
    arrays = {key: tensor.numpy() for key, tensor in array_tensors.items()}

    genome_end = int(arrays["node_end"].max())
    positive_regions = [] if npz_mode else build_positive_regions(args.labels, genome_end)
    val_regions, val_contact_images = [], None
    train_contact_images = test_contact_images = None
    hard_negative_count = 0
    balanced_positive_count = 0
    balanced_negative_count = 0
    if npz_mode:
        train_regions, train_contact_images = _load_npz_contact_bundle(
            (args.opcid_train_npz, args.chin_train_npz, args.chid_train_npz)
        )
        train_regions, train_contact_images, hard_negative_count = balance_npz_training_regions(
            train_regions, train_contact_images, args.seed, args.include_hard_negative
        )
        balanced_positive_count = sum(region[2].any() for region in train_regions)
        balanced_negative_count = sum(not region[2].any() for region in train_regions)
        if args.dev_earlystop or args.threshold_finetune:
            train_regions, train_contact_images, val_regions, val_contact_images = _split_npz_validation(
                train_regions, train_contact_images, args.validation_fraction, args.seed
            )
        test_regions, test_contact_images = _load_npz_contact_bundle(
            (args.opcid_test_npz, args.chin_test_npz, args.chid_test_npz)
        )
        test_array_tensors = load_graph_archive(args.test_input)
        if test_array_tensors["x"].shape[1] != array_tensors["x"].shape[1]:
            raise ValueError("Rep1 and rep2 graph archives have different node feature dimensions")
    else:
        regions, negative_region_count = build_training_regions(
            args.labels, genome_end, args.seed,
            args.include_background_negatives, args.include_hard_negative,
        )
        balanced_positive_count = sum(region[2].any() for region in regions)
        balanced_negative_count = sum(not region[2].any() for region in regions)
        if balanced_positive_count != balanced_negative_count:
            raise RuntimeError(
                f"Balanced pool invariant failed: positive={balanced_positive_count}, "
                f"negative={balanced_negative_count}"
            )
        if args.include_hard_negative:
            hard_negative_count = min(
                len(positive_regions), len(_hard_negative_candidates(positive_regions, genome_end))
            )
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(regions))
        split = max(1, int(round(len(regions) * (1 - args.test_fraction))))
        train_regions = [regions[i] for i in order[:split]]
        test_regions = [regions[i] for i in order[split:]]
        test_array_tensors = array_tensors
        negative_region_count = int(negative_region_count)
        if args.dev_earlystop or args.threshold_finetune:
            train_order = rng.permutation(len(train_regions))
            val_split = max(1, int(round(len(train_regions) * (1 - args.validation_fraction))))
            val_regions = [train_regions[i] for i in train_order[val_split:]]
            train_regions = [train_regions[i] for i in train_order[:val_split]]
    if npz_mode:
        negative_region_count = int(sum(not region[2].any() for region in train_regions))
    if not train_regions or not test_regions:
        raise ValueError("The train/test split produced an empty partition; adjust --test-fraction or add more labels.")
    train_positive_count = int(sum(region[2].any() for region in train_regions))
    train_negative_count = int(sum(not region[2].any() for region in train_regions))
    test_positive_count = int(sum(region[2].any() for region in test_regions))
    test_negative_count = int(sum(not region[2].any() for region in test_regions))
    train_label_positive_counts = np.stack([region[2] for region in train_regions]).sum(axis=0).astype(int).tolist()
    test_label_positive_counts = np.stack([region[2] for region in test_regions]).sum(axis=0).astype(int).tolist()

    model = SpecializedRegionModel(
        arrays["x"].shape[1], args.hidden_dim, args.embedding_dim, args.classifier_hidden_dim,
        enable_opcid_verifier=args.opcid_verifier,
    ).to(device)
    train_labels = np.stack([region[2] for region in train_regions])
    class_weights = torch.from_numpy(
        effective_number_weights(train_labels, args.class_balanced_beta)
    ).to(device)
    opcid_positive_count = float(train_labels[:, 0].sum())
    opcid_negative_count = float(len(train_labels) - opcid_positive_count)
    if opcid_positive_count <= 0:
        raise ValueError("The training split contains no OPCID positive labels")
    automatic_opcid_pos_weight = opcid_negative_count / opcid_positive_count
    opcid_pos_weight = (
        float(args.opcid_pos_weight)
        if args.opcid_pos_weight > 0
        else min(automatic_opcid_pos_weight, args.max_opcid_pos_weight)
    )
    positive_weights = torch.ones(3, dtype=torch.float32, device=device)
    positive_weights[0] = opcid_pos_weight
    verifier_pos_weight = torch.tensor(
        opcid_negative_count / opcid_positive_count,
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    pin_memory = bool(args.pin_memory and device.type == "cuda")
    persistent_workers = bool(args.persistent_workers and args.num_workers > 0)
    if not args.evaluate:
        print(
            json.dumps(
                {
                    "num_workers": args.num_workers,
                    "pin_memory": pin_memory,
                    "persistent_workers": persistent_workers,
                    "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
                    "train_regions": len(train_regions),
                    "test_regions": len(test_regions),
                    "train_positive": train_positive_count,
                    "train_negative": train_negative_count,
                    "test_positive": test_positive_count,
                    "test_negative": test_negative_count,
                    "train_label_positive_counts": train_label_positive_counts,
                    "test_label_positive_counts": test_label_positive_counts,
                    "opcid_positive_count": int(opcid_positive_count),
                    "opcid_negative_count": int(opcid_negative_count),
                    "opcid_pos_weight": opcid_pos_weight,
                    "opcid_pos_weight_source": "argument" if args.opcid_pos_weight > 0 else "auto_negative_over_positive",
                    "negative_regions": negative_region_count,
                    "balanced_pool_positive": balanced_positive_count,
                    "balanced_pool_negative": balanced_negative_count,
                    "sample_mode": "current_dataset_rep1_rep2_balanced" if npz_mode else "cooler_custom_split_balanced",
                    "input_mode": "current-dataset" if npz_mode else "cooler",
                    "device": str(device),
                },
                ensure_ascii=False,
                default=_json_default,
            )
        )

    train_dataset = RegionDataset(
        train_regions, array_tensors, args.cool_input, args.patch_size,
        train_contact_images if npz_mode else None,
    )
    val_dataset = RegionDataset(
        val_regions, array_tensors, args.cool_input, args.patch_size,
        val_contact_images if npz_mode else None,
    ) if val_regions else None
    test_dataset = RegionDataset(
        test_regions, test_array_tensors, args.cool_input, args.patch_size,
        test_contact_images if npz_mode else None,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = _make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor,
        generator=loader_generator,
    )
    val_loader = _make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor,
    ) if val_dataset is not None else None
    non_blocking = pin_memory and device.type == "cuda"

    def run_loader(loader, *, training=False):
        total_loss, steps = 0.0, 0
        collected_logits, collected_embeddings, collected_labels, collected_verifier_logits = [], [], [], []
        for batch in loader:
            if not batch:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            batch_logits, batch_embeddings, batch_labels, batch_verifier_logits = [], [], [], []
            for sample in batch:
                graph, patches, label = _move_sample_to_device(sample, device, non_blocking)
                opcid, chin, chid = model.encode(graph, patches, augment_opcid=training)
                logits = torch.stack((model.opcid_head(opcid), model.chin_head(chin), model.chid_head(chid)))
                batch_logits.append(logits)
                batch_embeddings.append((opcid, chin, chid))
                batch_labels.append(label)
                if model.opcid_verifier is not None:
                    batch_verifier_logits.append(model.verifier_logits(patches, augment=training))
            logits_tensor = torch.stack(batch_logits)
            labels_tensor = torch.stack(batch_labels)
            branch_embeddings = [torch.stack([item[index] for item in batch_embeddings]) for index in range(3)]
            per_class_losses = asymmetric_loss_per_class(
                logits_tensor, labels_tensor, class_weights,
                positive_weights,
                args.asl_gamma_pos, args.asl_gamma_neg, args.asl_clip,
            )
            probabilities = torch.sigmoid(logits_tensor)
            hierarchy_loss = F.relu(probabilities[:, 2] - probabilities[:, 1]).pow(2).mean()
            branch_losses = []
            for class_index, embedding in enumerate(branch_embeddings):
                one_class_labels = labels_tensor[:, class_index:class_index + 1]
                contrastive = supervised_contrastive_loss(embedding, one_class_labels, args.supcon_temperature)
                centers = center_loss(embedding, one_class_labels)
                branch_losses.append(
                    per_class_losses[class_index]
                    + args.supcon_loss_weight * contrastive
                    + args.center_loss_weight * centers
                )
            loss = branch_losses[0] + branch_losses[1] + branch_losses[2] + args.hier_loss_weight * hierarchy_loss
            if model.opcid_verifier is not None:
                verifier_logits_tensor = torch.stack(batch_verifier_logits)
                verifier_targets = labels_tensor[:, 0]
                verifier_loss = F.binary_cross_entropy_with_logits(
                    verifier_logits_tensor, verifier_targets, pos_weight=verifier_pos_weight
                )
                loss = loss + verifier_loss
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.detach().cpu())
            steps += 1
            if not training:
                collected_logits.extend([row.detach().cpu() for row in batch_logits])
                collected_embeddings.extend([
                    torch.cat([item.detach().cpu() for item in embeddings])
                    for embeddings in batch_embeddings
                ])
                collected_labels.extend([row.detach().cpu().numpy() for row in batch_labels])
                collected_verifier_logits.extend([row.detach().cpu() for row in batch_verifier_logits])
        return total_loss / max(steps, 1), collected_logits, collected_embeddings, collected_labels, collected_verifier_logits

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, _, _, _, _ = run_loader(train_loader, training=True)
        val_loss = None
        if val_loader is not None and (args.dev_earlystop or args.threshold_finetune):
            model.eval()
            with torch.no_grad():
                val_loss, _, _, _, _ = run_loader(val_loader)
            if args.dev_earlystop and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                epochs_without_improvement = 0
            elif args.dev_earlystop:
                epochs_without_improvement += 1
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or (args.dev_earlystop and epochs_without_improvement >= args.earlystop_patience):
            suffix = f" val_loss={val_loss:.5f}" if val_loss is not None else ""
            print(f"region epoch {epoch:03d}/{args.epochs} loss={epoch_loss:.5f}{suffix}")
        if args.dev_earlystop and epochs_without_improvement >= args.earlystop_patience:
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}, val_loss={best_val_loss:.5f}")
            break
    if args.dev_earlystop and best_state is not None:
        model.load_state_dict(best_state)

    val_logits, val_y, val_verifier_logits = [], [], []
    if val_loader is not None:
        model.eval()
        with torch.no_grad():
            _, val_logits, _, val_y, val_verifier_logits = run_loader(val_loader)
    if (args.dev_earlystop or args.threshold_finetune) and not val_logits:
        raise RuntimeError("The rep1 validation split produced no usable graph/contact samples")

    # Stop the persistent training workers before creating the evaluation
    # loader, so the two worker pools do not double peak memory usage.
    del train_loader
    del train_dataset
    if val_loader is not None:
        del val_loader
    if val_dataset is not None:
        del val_dataset
    gc.collect()

    model.eval()
    test_loader = _make_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    train_emb, train_logits, train_y, train_verifier_logits = [], [], [], []
    test_emb, test_logits, test_y, test_coords, test_verifier_logits = [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            for sample in batch:
                graph, patches, _ = _move_sample_to_device(sample, device, non_blocking)
                opcid, chin, chid = model.encode(graph, patches)
                test_logits.append(torch.stack((model.opcid_head(opcid), model.chin_head(chin), model.chid_head(chid))).cpu())
                test_emb.append(torch.cat((opcid, chin, chid)).cpu())
                if model.opcid_verifier is not None:
                    test_verifier_logits.append(model.verifier_logits(patches).cpu())
                region = test_regions[sample["index"]]
                test_y.append(region[2])
                test_coords.append((region[0], region[1]))
    # Train embeddings are used only to form known-class prototypes for novelty detection.
    model.eval()
    prototype_dataset = RegionDataset(
        train_regions, array_tensors, args.cool_input, args.patch_size,
        train_contact_images if npz_mode else None,
    )
    prototype_loader = _make_loader(
        prototype_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, prefetch_factor=args.prefetch_factor,
    )
    with torch.no_grad():
        for batch in prototype_loader:
            for sample in batch:
                graph, patches, _ = _move_sample_to_device(sample, device, non_blocking)
                opcid, chin, chid = model.encode(graph, patches)
                train_emb.append(torch.cat((opcid, chin, chid)).cpu())
                train_logits.append(torch.stack((model.opcid_head(opcid), model.chin_head(chin), model.chid_head(chid))).cpu())
                train_y.append(train_regions[sample["index"]][2])
                if model.opcid_verifier is not None:
                    train_verifier_logits.append(model.verifier_logits(patches).cpu())
    del prototype_loader, prototype_dataset, test_loader, test_dataset
    gc.collect()
    if not train_emb or not test_emb:
        raise RuntimeError(
            f"No usable region embeddings were produced (train={len(train_emb)}, test={len(test_emb)}). "
            "Check that graph coordinates and .cool bins use the same chromosome/resolution."
        )

    thresholds = [args.opcid_threshold, args.chin_threshold, args.chid_threshold]
    verifier_threshold = args.opcid_verifier_threshold
    threshold_selection = {}
    if args.threshold_finetune:
        if args.opcid_verifier and val_verifier_logits:
            (
                thresholds[0], verifier_threshold, precision, recall, f1, selection_status
            ) = _select_opcid_cascade_thresholds(
                val_logits,
                val_verifier_logits,
                np.asarray(val_y)[:, 0],
                args.opcid_min_recall,
                (thresholds[0], verifier_threshold),
            )
            threshold_selection["OPCID"] = {
                "threshold": thresholds[0],
                "verifier_threshold": verifier_threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "minimum_recall": args.opcid_min_recall,
                **selection_status,
            }
        else:
            chosen, precision, recall, f1, selection_status = _select_threshold_by_precision(
                [row[0:1] for row in val_logits],
                np.asarray(val_y)[:, 0], args.opcid_min_recall, thresholds[0],
            )
            thresholds[0] = chosen
            threshold_selection["OPCID"] = {
                "threshold": chosen,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "minimum_recall": args.opcid_min_recall,
                **selection_status,
            }
        for class_index, name, minimum_recall in (
            (1, "CHIN", args.chin_min_recall),
            (2, "CHID", args.chid_min_recall),
        ):
            chosen, precision, recall, f1, selection_status = _select_threshold_by_precision(
                [row[class_index:class_index + 1] for row in val_logits],
                np.asarray(val_y)[:, class_index], minimum_recall, thresholds[class_index],
            )
            thresholds[class_index] = chosen
            threshold_selection[name] = {
                "threshold": chosen,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "minimum_recall": minimum_recall,
                **selection_status,
            }
    thresholds = tuple(thresholds)
    prototype_state = fit_embedding_prototypes(train_emb, train_y)
    test_novelty_distance = embedding_novelty_distance(test_emb, prototype_state)
    stats = {
        "train": evaluate_with_thresholds(train_logits, train_y, thresholds, train_verifier_logits if args.opcid_verifier else None, verifier_threshold if args.opcid_verifier else None),
        "validation": evaluate_with_thresholds(val_logits, val_y, thresholds, val_verifier_logits if args.opcid_verifier and val_logits else None, verifier_threshold if args.opcid_verifier and val_logits else None) if val_logits else None,
        "test": evaluate_with_thresholds(test_logits, test_y, thresholds, test_verifier_logits if args.opcid_verifier else None, verifier_threshold if args.opcid_verifier else None),
        "train_regions": len(train_emb),
        "test_regions": len(test_emb),
        "train_positive": train_positive_count,
        "train_negative": train_negative_count,
        "test_positive": test_positive_count,
        "test_negative": test_negative_count,
        "train_label_positive_counts": train_label_positive_counts,
        "test_label_positive_counts": test_label_positive_counts,
        "opcid_positive_count": int(opcid_positive_count),
        "opcid_negative_count": int(opcid_negative_count),
        "opcid_pos_weight": opcid_pos_weight,
        "opcid_pos_weight_source": "argument" if args.opcid_pos_weight > 0 else "auto_negative_over_positive",
        "opcid_verifier": {"enabled": args.opcid_verifier, "threshold": verifier_threshold},
        "negative_regions": negative_region_count,
        "balanced_pool_positive": balanced_positive_count,
        "balanced_pool_negative": balanced_negative_count,
        "validation_regions": len(val_regions),
        "early_stopping": {"enabled": args.dev_earlystop, "best_epoch": best_epoch, "best_validation_loss": best_val_loss if best_epoch else None},
        "threshold_selection": threshold_selection,
        "hard_negative_regions": hard_negative_count,
        "novelty_distance_threshold": args.novelty_distance_threshold,
        "sample_mode": "current_dataset_rep1_rep2_balanced" if npz_mode else "cooler_custom_split_balanced",
        "input_mode": "current-dataset" if npz_mode else "cooler",
        "region_feature_dim": 3 * 8,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "classifier_hidden_dim": args.classifier_hidden_dim,
        "class_thresholds": list(thresholds),
        "context": args.context,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    torch.save(
        {
            "gnn_encoder": model.state_dict(),
            "specialized_model": model.state_dict(),
            "args": vars(args),
            "class_thresholds": thresholds,
            "prototype_state": prototype_state,
            "best_epoch": best_epoch,
            "opcid_pos_weight": opcid_pos_weight,
            "opcid_verifier_threshold": verifier_threshold,
        },
        args.output_dir / f"{stem}_region_model.pt",
    )
    np.savez_compressed(
        args.output_dir / f"{stem}_region_test.npz",
        embedding=torch.stack(test_emb).numpy(),
        labels=np.asarray(test_y),
        novelty_distance=test_novelty_distance,
        opcid_verifier_probability=(
            torch.sigmoid(torch.stack(test_verifier_logits)).numpy()
            if test_verifier_logits else np.zeros(len(test_y), dtype=np.float32)
        ),
        start=np.asarray([coord[0] for coord in test_coords]),
        end=np.asarray([coord[1] for coord in test_coords]),
        metadata=np.array(json.dumps(stats, default=_json_default)),
    )

    if args.scan:
        positive_lengths = [end - start for start, end, _ in positive_regions]
        window = args.scan_window or int(np.median(positive_lengths))
        edge_index = arrays["edge_index"]
        edge_weight = arrays["edge_weight"]
        scan_rows = []
        cool = cooler.Cooler(str(args.cool_input))
        cool_matrix = cool.matrix(balance=False, sparse=False)
        with torch.no_grad():
            for start in range(0, max(1, genome_end - window + 1), args.scan_step):
                end = min(genome_end, start + window)
                graph_start, graph_end = graph_context_bounds(start, end, arrays, 500)
                graph = region_graph(arrays, edge_index, edge_weight, graph_start, graph_end, device)
                if graph is None:
                    continue
                patches = multiscale_patches(
                    cool_matrix,
                    arrays,
                    (start, end),
                    (250, 500, 1000),
                    args.patch_size,
                    device,
                )
                if patches is None:
                    continue
                opcid, chin, chid = model.encode(graph, patches)
                logits = torch.stack((model.opcid_head(opcid), model.chin_head(chin), model.chid_head(chid)))
                probability = torch.sigmoid(logits).cpu().numpy()
                verifier_probability = float(torch.sigmoid(model.verifier_logits(patches)).cpu()) if model.opcid_verifier is not None else float("nan")
                if model.opcid_verifier is not None and verifier_probability < verifier_threshold:
                    probability[0] = 0.0
                novelty_distance = float(embedding_novelty_distance(
                    [torch.cat((opcid, chin, chid)).cpu()], prototype_state
                )[0])
                predicted_known = bool(np.any(probability >= np.asarray(thresholds)))
                known = any(
                    start < positive_end and end > positive_start
                    for positive_start, positive_end, _ in positive_regions
                )
                scan_rows.append(
                    (
                        start,
                        end,
                        *probability.tolist(),
                        verifier_probability,
                        novelty_distance,
                        bool((not known) and (not predicted_known) and novelty_distance >= args.novelty_distance_threshold),
                    )
                )
        scan_dtype = [
            ("start", "i8"),
            ("end", "i8"),
            ("p_opcid", "f4"),
            ("p_chin", "f4"),
            ("p_chid", "f4"),
            ("p_opcid_verifier", "f4"),
            ("novelty_distance", "f4"),
            ("novel_candidate", "?"),
        ]
        np.save(args.output_dir / f"{stem}_whole_genome_scan.npy", np.asarray(scan_rows, dtype=scan_dtype))
        stats["scan_windows"] = len(scan_rows)
        stats["scan_novel_candidates"] = int(sum(row[-1] for row in scan_rows))
        if not args.evaluate:
            print(
                json.dumps(
                    {
                        "whole_genome_scan": stats["scan_windows"],
                        "novel_candidates": stats["scan_novel_candidates"],
                    },
                    ensure_ascii=False,
                    default=_json_default,
                )
            )

    if args.evaluate:
        evaluation_dir = args.evaluate_output_dir or (args.output_dir / "evaluation")
        report = _write_evaluation_report(stats["test"], evaluation_dir, stem)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(json.dumps(stats, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
