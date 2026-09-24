#!/usr/bin/env python3
"""Build a sparse top-k contact graph from one or more Cooler files.

The output is a compressed NumPy ``.npz`` archive. It can be loaded without
PyTorch using ``numpy.load(..., allow_pickle=False)`` and converted to the
``x``, ``edge_index`` and ``edge_weight`` tensors expected by most GNN stacks.

Example (from the repository root)::

    python data/datasets/graph_parse.py --topk 20

By default, every ``*.cool`` file beside this script is processed and graphs
are written to ``data/graphs``. Rows of the contact matrix are read in chunks;
the full dense matrix is never materialized.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_dependencies() -> tuple[Any, Any]:
    """Import the optional data dependencies with an actionable error."""
    try:
        import cooler
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "graph_parse.py requires the 'cooler' package and its dependencies. "
            "Install them with: python -m pip install cooler"
        ) from exc
    return cooler, np


def _normalized_log1p(values: Any, np: Any) -> Any:
    """Return log1p(values) scaled to [0, 1], handling all-zero vectors."""
    values = np.log1p(np.maximum(values, 0)).astype(np.float32, copy=False)
    maximum = float(values.max()) if values.size else 0.0
    if maximum > 0:
        values /= maximum
    return values


def build_graph(
    cool_path: Path,
    output_path: Path,
    *,
    topk: int,
    chunk_size: int,
    min_count: float,
) -> None:
    """Convert one Cooler contact map to an undirected top-k graph archive."""
    cooler, np = _load_dependencies()

    cool = cooler.Cooler(str(cool_path))
    bins = cool.bins()[:]
    node_count = len(bins)
    if node_count == 0:
        raise ValueError(f"No genomic bins found in {cool_path}")

    chrom = np.asarray(bins["chrom"].astype(str).to_numpy(), dtype=str)
    starts = bins["start"].to_numpy(dtype=np.int64, copy=True)
    ends = bins["end"].to_numpy(dtype=np.int64, copy=True)
    chrom_names, chrom_ids = np.unique(chrom, return_inverse=True)

    # Genomic position is represented relative to the chromosome length.
    sizes = {str(name): int(length) for name, length in cool.chromsizes.items()}
    chrom_lengths = np.fromiter(
        (sizes[str(name)] for name in chrom_names), dtype=np.float64,
        count=len(chrom_names),
    )
    relative_position = (
        (starts.astype(np.float64) + ends.astype(np.float64))
        / (2.0 * chrom_lengths[chrom_ids])
    ).astype(np.float32)
    np.clip(relative_position, 0.0, 1.0, out=relative_position)

    # Cooler exposes the symmetric sparse matrix. Slice it by row so memory
    # usage is proportional to a chunk of observed contacts, not N squared.
    matrix = cool.matrix(balance=False, sparse=True)
    row_totals = np.zeros(node_count, dtype=np.float64)
    topk_totals = np.zeros(node_count, dtype=np.float64)
    source_parts = []
    target_parts = []
    weight_parts = []

    for row_start in range(0, node_count, chunk_size):
        row_end = min(row_start + chunk_size, node_count)
        block = matrix[row_start:row_end, :].tocsr()
        row_totals[row_start:row_end] = np.asarray(block.sum(axis=1)).reshape(-1)

        # At most ``chunk_size * topk`` entries are retained for each chunk.
        max_entries = (row_end - row_start) * topk
        chunk_sources = np.empty(max_entries, dtype=np.int64)
        chunk_targets = np.empty(max_entries, dtype=np.int64)
        chunk_weights = np.empty(max_entries, dtype=np.float32)
        used = 0

        for local_row in range(row_end - row_start):
            row = row_start + local_row
            begin, finish = block.indptr[local_row : local_row + 2]
            columns = block.indices[begin:finish]
            values = block.data[begin:finish]
            valid = (
                (columns != row)
                & np.isfinite(values)
                & (values > 0)
                & (values >= min_count)
            )
            candidate_columns = columns[valid]
            candidate_values = values[valid]
            if candidate_values.size == 0:
                continue

            if candidate_values.size > topk:
                chosen = np.argpartition(candidate_values, -topk)[-topk:]
                # Keep output order deterministic among the selected entries.
                chosen = chosen[np.lexsort((candidate_columns[chosen], -candidate_values[chosen]))]
            else:
                chosen = np.lexsort((candidate_columns, -candidate_values))

            selected_columns = candidate_columns[chosen]
            selected_values = candidate_values[chosen]
            count = selected_values.size
            chunk_sources[used : used + count] = row
            chunk_targets[used : used + count] = selected_columns
            chunk_weights[used : used + count] = selected_values
            used += count
            topk_totals[row] = selected_values.sum(dtype=np.float64)

        if used:
            source_parts.append(chunk_sources[:used].copy())
            target_parts.append(chunk_targets[:used].copy())
            weight_parts.append(chunk_weights[:used].copy())

        print(
            f"[{cool_path.name}] scanned bins {row_start:,}-{row_end:,} "
            f"of {node_count:,}",
            file=sys.stderr,
        )

    if source_parts:
        source = np.concatenate(source_parts)
        target = np.concatenate(target_parts)
        selected_weight = np.concatenate(weight_parts)

        # Make the graph undirected: a pair is included if either endpoint
        # selected the other. The resulting degree can therefore exceed topk.
        left = np.minimum(source, target)
        right = np.maximum(source, target)
        order = np.lexsort((right, left))
        sorted_left = left[order]
        sorted_right = right[order]
        sorted_weight = selected_weight[order]
        first_of_pair = np.empty(sorted_left.size, dtype=bool)
        first_of_pair[0] = True
        first_of_pair[1:] = (sorted_left[1:] != sorted_left[:-1]) | (
            sorted_right[1:] != sorted_right[:-1]
        )
        pair_starts = np.flatnonzero(first_of_pair)
        pair_left = sorted_left[pair_starts]
        pair_right = sorted_right[pair_starts]
        pair_weight = np.maximum.reduceat(sorted_weight, pair_starts)

        edge_index = np.empty((2, pair_left.size * 2), dtype=np.int64)
        edge_index[0, : pair_left.size] = pair_left
        edge_index[1, : pair_left.size] = pair_right
        edge_index[0, pair_left.size :] = pair_right
        edge_index[1, pair_left.size :] = pair_left
        edge_weight = np.concatenate((pair_weight, pair_weight)).astype(np.float32, copy=False)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_weight = np.empty((0,), dtype=np.float32)

    x = np.column_stack(
        (
            relative_position,
            _normalized_log1p(row_totals, np),
            _normalized_log1p(topk_totals, np),
        )
    ).astype(np.float32, copy=False)

    metadata = {
        "schema_version": 1,
        "source_cool": cool_path.name,
        "node_count": int(node_count),
        "directed_edge_count": int(edge_index.shape[1]),
        "topk": int(topk),
        "min_count_inclusive_floor": float(min_count),
        "edge_semantics": "symmetric; union of each node's top-k positive contacts",
        "edge_weight_semantics": "raw Cooler contact count",
        "node_feature_names": [
            "relative_chromosome_position",
            "normalized_log1p_total_contacts",
            "normalized_log1p_topk_contact_sum",
        ],
        "labels": "not included; attach task-specific labels before supervised training",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        node_chrom=chrom,
        node_chrom_id=chrom_ids.astype(np.int32, copy=False),
        node_start=starts,
        node_end=ends,
        chrom_names=chrom_names.astype(str),
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    print(
        f"Saved {output_path} ({node_count:,} nodes, "
        f"{edge_index.shape[1]:,} directed edges)",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Parse Cooler contact maps into sparse top-k GNN graph files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=script_dir,
        help="A .cool file or directory (default: directory containing this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir.parent / "graphs",
        help="Directory for generated .npz graph files (default: data/graphs).",
    )
    parser.add_argument(
        "--topk", type=int, default=20,
        help="Keep up to this many strongest non-self contacts per bin (default: 20).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=4096,
        help="Number of matrix rows read at a time (default: 4096).",
    )
    parser.add_argument(
        "--min-count", type=float, default=0.0,
        help="Drop contacts below this count; zero-valued contacts are always ignored.",
    )
    args = parser.parse_args()
    if args.topk <= 0:
        parser.error("--topk must be greater than zero")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be greater than zero")
    if args.min_count < 0:
        parser.error("--min-count cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input path does not exist: {args.input}")

    if args.input.is_file():
        inputs = [args.input]
    else:
        inputs = sorted(path for path in args.input.rglob("*.cool") if path.is_file())
    if not inputs:
        raise SystemExit(f"No .cool files found under: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cool_path in inputs:
        output_path = args.output_dir / f"{cool_path.stem}_top{args.topk}.npz"
        build_graph(
            cool_path,
            output_path,
            topk=args.topk,
            chunk_size=args.chunk_size,
            min_count=args.min_count,
        )


if __name__ == "__main__":
    main()
