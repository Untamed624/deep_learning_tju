"""Region-level Hi-C GNN/CNN training with parallel CPU data loading.

This keeps the model, labels, losses, evaluation, and scan behavior of
``region_GNN.py`` while training only on annotated regions. Random background
negative regions are intentionally not generated. Region graphs and Cooler
contact patches are prepared by DataLoader workers on CPU; the main process
transfers each sample to the selected device for forward/backward computation.
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

try:
    from .GNN import GraphModel, MultiLabelClassifier, read_interval_labels
    from .region_GNN import (
        ContactCNN,
        CrossModalFusion,
        MultiScaleFusion,
        fused_region_embedding,
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


def build_training_regions(labels_dir: Path, genome_end: int, seed: int, include_background_negatives: bool):
    """Build either annotated-only or annotated-plus-background training data."""
    positives = build_positive_regions(labels_dir, genome_end)
    if not include_background_negatives:
        return positives, 0

    rng = np.random.default_rng(seed)
    occupied = [(start, end) for start, end, _ in positives]
    negatives = []
    attempts = 0
    while len(negatives) < len(positives) and attempts < max(1, len(positives) * 1000):
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
    return positives + negatives, len(negatives)


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


class SpecializedRegionModel(torch.nn.Module):
    """Class-specific architecture for square, diagonal, and cluster signals."""

    def __init__(self, node_dim, hidden_dim, embedding_dim, classifier_hidden_dim):
        super().__init__()
        self.gnn = GraphModel(node_dim, hidden_dim, embedding_dim)
        self.opcid_cnn = torch.nn.ModuleList([ContactCNN(embedding_dim) for _ in (250, 500, 1000)])
        self.chin_cnn = torch.nn.ModuleList([ContactCNN(embedding_dim) for _ in (250, 500, 1000)])
        self.cross_attention = CrossModalFusion(embedding_dim, 64, embedding_dim)
        self.embedding_dim = embedding_dim
        self.opcid_head = BinaryHead(embedding_dim, classifier_hidden_dim)
        # CrossModalFusion returns 2*embedding_dim; append graph topology and
        # diagonal statistics for CHIN, then cluster statistics for CHID.
        self.chin_head = BinaryHead(2 * embedding_dim + 2 + 24, classifier_hidden_dim)
        self.chid_head = BinaryHead(2 * embedding_dim + 2 + 24 + 18, classifier_hidden_dim)

    def encode(self, graph, patches):
        z = self.gnn(graph[0], graph[1], graph[2])
        topology = torch.log1p(torch.tensor([graph[0].size(0), graph[1].size(1)], dtype=z.dtype, device=z.device))
        # Keep a batch dimension for Conv2d/BatchNorm2d.  Each scale contains
        # [views=3, channels=3, H, W], so the square view is [1, 3, H, W].
        opcid = torch.stack(
            [branch(patch_views[0:1]).squeeze(0) for branch, patch_views in zip(self.opcid_cnn, patches)]
        ).mean(dim=0)
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


def evaluate_with_thresholds(logits, labels, thresholds):
    """Evaluate each class with its own decision threshold."""
    probability = torch.sigmoid(torch.stack(logits)).detach().cpu().numpy()
    labels = np.asarray(labels, dtype=np.float32)
    threshold_array = np.asarray(thresholds, dtype=np.float32).reshape(1, 3)
    prediction = (probability >= threshold_array).astype(np.float32)
    tp = (prediction * labels).sum(axis=0)
    precision = tp / np.maximum(prediction.sum(axis=0), 1.0)
    recall = tp / np.maximum(labels.sum(axis=0), 1.0)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "binary_accuracy": float((prediction == labels).mean()),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "thresholds": threshold_array.reshape(-1).tolist(),
    }


class RegionDataset(Dataset):
    """Prepare variable-size local graphs and multiscale Cooler patches."""

    def __init__(self, regions, shared_arrays, cool_path: Path, patch_size: int):
        self.regions = regions
        # CPU tensors are shared once when workers are enabled. This avoids
        # serializing a full graph copy into every Windows worker process.
        self.shared_arrays = shared_arrays
        self.cool_path = str(cool_path)
        self.patch_size = patch_size
        self._pid = None
        self._arrays = None
        self._cool_matrix = None
        self._cooler = None

    def __len__(self):
        return len(self.regions)

    def _init_process_resources(self):
        pid = os.getpid()
        if self._pid == pid and self._arrays is not None and self._cool_matrix is not None:
            return

        # Do not open an HDF5/Cooler handle in the parent and pass it to child
        # processes. Every worker opens its own handle after it starts.
        import cooler

        self._arrays = {key: tensor.numpy() for key, tensor in self.shared_arrays.items()}
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

        patches = multiscale_patches(
            self._cool_matrix,
            arrays,
            (start, end),
            (250, 500, 1000),
            self.patch_size,
            torch.device("cpu"),
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
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--cnn-dim", type=int, default=128, help="Per-scale CNN token projection width (kept for compatibility).")
    parser.add_argument("--patch-size", type=int, default=32, help="Fixed CNN patch size; 32 reduces overfitting and attention cost for small datasets.")
    parser.add_argument("--classifier-hidden-dim", type=int, default=32)
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
    parser.add_argument("--chin-threshold", type=float, default=0.50, help="CHIN threshold.")
    parser.add_argument("--chid-threshold", type=float, default=0.40)
    parser.add_argument("--novelty-threshold", type=float, default=0.55)
    parser.add_argument(
        "--include-background-negatives",
        action="store_true",
        help="Add an equal-sized set of random non-overlapping background regions. Default: annotated regions only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scan", action="store_true", help="After training, scan the whole chromosome.")
    parser.add_argument("--scan-window", type=int, default=0, help="Window size in bp; 0 uses the median positive region size.")
    parser.add_argument("--scan-step", type=int, default=500)
    args = parser.parse_args()
    if args.context < 0 or args.batch_size <= 0 or args.epochs <= 0:
        raise SystemExit("context, batch-size and epochs must be valid positive values")
    if args.num_workers < 0 or args.prefetch_factor <= 0:
        raise SystemExit("num-workers must be >= 0 and prefetch-factor must be > 0")
    if args.embedding_dim <= 0 or args.embedding_dim % 4 != 0:
        raise SystemExit("embedding-dim must be a positive multiple of 4 because cross-attention uses 4 heads")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but this PyTorch build/runtime has no available CUDA device.")

    try:
        import cooler  # noqa: F401 - fail early with a useful message
    except ImportError as exc:
        raise SystemExit("CNN matrix input requires cooler; install it in the active environment.") from exc

    # Share the arrays needed by workers rather than embedding a large NumPy
    # graph in each spawned Windows process. Only these keys are used here.
    array_tensors = {}
    with np.load(args.input, allow_pickle=False) as archive:
        for key in ("x", "node_start", "node_end", "edge_index", "edge_weight"):
            if key not in archive:
                raise ValueError(f"Graph archive is missing required array: {key}")
            tensor = torch.from_numpy(np.ascontiguousarray(archive[key]))
            if args.num_workers > 0:
                try:
                    tensor.share_memory_()
                except RuntimeError as exc:
                    raise SystemExit(
                        "Could not place graph arrays in shared memory for DataLoader workers. "
                        "Try --num-workers 0 or reduce graph size."
                    ) from exc
            array_tensors[key] = tensor
    arrays = {key: tensor.numpy() for key, tensor in array_tensors.items()}

    genome_end = int(arrays["node_end"].max())
    regions, negative_region_count = build_training_regions(
        args.labels,
        genome_end,
        args.seed,
        args.include_background_negatives,
    )
    positive_regions = build_positive_regions(args.labels, genome_end)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(regions))
    split = max(1, int(round(len(regions) * (1 - args.test_fraction))))
    train_regions = [regions[i] for i in order[:split]]
    test_regions = [regions[i] for i in order[split:]]
    if not train_regions or not test_regions:
        raise ValueError("The train/test split produced an empty partition; adjust --test-fraction or add more labels.")

    model = SpecializedRegionModel(
        arrays["x"].shape[1], args.hidden_dim, args.embedding_dim, args.classifier_hidden_dim
    ).to(device)
    train_labels = np.stack([region[2] for region in train_regions])
    positive = torch.from_numpy(train_labels.sum(axis=0)).to(device).clamp_min(1.0)
    pos_weight = ((len(train_regions) - positive).clamp_min(1.0) / positive).float()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    pin_memory = bool(args.pin_memory and device.type == "cuda")
    persistent_workers = bool(args.persistent_workers and args.num_workers > 0)
    print(
        json.dumps(
            {
                "num_workers": args.num_workers,
                "pin_memory": pin_memory,
                "persistent_workers": persistent_workers,
                "prefetch_factor": args.prefetch_factor if args.num_workers > 0 else None,
                "train_regions": len(train_regions),
                "test_regions": len(test_regions),
                "negative_regions": negative_region_count,
                "sample_mode": (
                    "annotated_plus_random_background"
                    if args.include_background_negatives
                    else "annotated_regions_only"
                ),
                "device": str(device),
            },
            ensure_ascii=False,
        )
    )

    train_dataset = RegionDataset(train_regions, array_tensors, args.cool_input, args.patch_size)
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
    non_blocking = pin_memory and device.type == "cuda"

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.train()
        epoch_loss = 0.0
        optimizer_steps = 0
        for batch in train_loader:
            if not batch:
                continue
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for sample in batch:
                graph, patches, label = _move_sample_to_device(sample, device, non_blocking)
                logits = model(graph, patches)
                classification_loss = F.binary_cross_entropy_with_logits(logits, label, pos_weight=pos_weight)
                probabilities = torch.sigmoid(logits)
                hierarchy_loss = F.relu(probabilities[2] - probabilities[1]).pow(2)
                losses.append(classification_loss + 0.10 * hierarchy_loss)
            if losses:
                loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                optimizer_steps += 1
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(
                f"region epoch {epoch:03d}/{args.epochs} "
                f"loss={epoch_loss:.5f} optimizer_steps={optimizer_steps}"
            )

    # Stop the persistent training workers before creating the evaluation
    # loader, so the two worker pools do not double peak memory usage.
    del train_loader
    del train_dataset
    gc.collect()

    model.eval()
    model.eval()
    eval_regions = train_regions + test_regions
    eval_dataset = RegionDataset(eval_regions, array_tensors, args.cool_input, args.patch_size)
    eval_loader = _make_loader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )
    train_emb, train_logits, train_y, train_coords = [], [], [], []
    test_emb, test_logits, test_y, test_coords = [], [], [], []
    with torch.no_grad():
        for batch in eval_loader:
            for sample in batch:
                graph, patches, _ = _move_sample_to_device(sample, device, non_blocking)
                opcid, chin, chid = model.encode(graph, patches)
                logits = torch.stack((model.opcid_head(opcid), model.chin_head(chin), model.chid_head(chid)))
                embedding = torch.cat((opcid, chin, chid)).cpu()
                logits = logits.cpu()
                index = sample["index"]
                region = eval_regions[index]
                label = region[2]
                if index < len(train_regions):
                    train_emb.append(embedding)
                    train_logits.append(logits)
                    train_y.append(label)
                    train_coords.append((region[0], region[1]))
                else:
                    test_emb.append(embedding)
                    test_logits.append(logits)
                    test_y.append(label)
                    test_coords.append((region[0], region[1]))
    if not train_emb or not test_emb:
        raise RuntimeError(
            f"No usable region embeddings were produced (train={len(train_emb)}, test={len(test_emb)}). "
            "Check that graph coordinates and .cool bins use the same chromosome/resolution."
        )

    thresholds = (args.opcid_threshold, args.chin_threshold, args.chid_threshold)
    stats = {
        "train": evaluate_with_thresholds(train_logits, train_y, thresholds),
        "test": evaluate_with_thresholds(test_logits, test_y, thresholds),
        "train_regions": len(train_emb),
        "test_regions": len(test_emb),
        "negative_regions": negative_region_count,
        "sample_mode": (
            "annotated_plus_random_background"
            if args.include_background_negatives
            else "annotated_regions_only"
        ),
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
        },
        args.output_dir / f"{stem}_region_model.pt",
    )
    np.savez_compressed(
        args.output_dir / f"{stem}_region_test.npz",
        embedding=torch.stack(test_emb).numpy(),
        labels=np.asarray(test_y),
        start=np.asarray([coord[0] for coord in test_coords]),
        end=np.asarray([coord[1] for coord in test_coords]),
        metadata=np.array(json.dumps(stats)),
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
                logits = model(graph, patches)
                probability = torch.sigmoid(logits).cpu().numpy()
                max_probability = float(probability.max())
                known = any(
                    start < positive_end and end > positive_start
                    for positive_start, positive_end, _ in positive_regions
                )
                scan_rows.append(
                    (
                        start,
                        end,
                        *probability.tolist(),
                        1.0 - max_probability,
                        bool((not known) and max_probability < args.novelty_threshold),
                    )
                )
        scan_dtype = [
            ("start", "i8"),
            ("end", "i8"),
            ("p_opcid", "f4"),
            ("p_chin", "f4"),
            ("p_chid", "f4"),
            ("novelty_score", "f4"),
            ("novel_candidate", "?"),
        ]
        np.save(args.output_dir / f"{stem}_whole_genome_scan.npy", np.asarray(scan_rows, dtype=scan_dtype))
        stats["scan_windows"] = len(scan_rows)
        stats["scan_novel_candidates"] = int(sum(row[-1] for row in scan_rows))
        print(
            json.dumps(
                {
                    "whole_genome_scan": stats["scan_windows"],
                    "novel_candidates": stats["scan_novel_candidates"],
                },
                ensure_ascii=False,
            )
        )

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
