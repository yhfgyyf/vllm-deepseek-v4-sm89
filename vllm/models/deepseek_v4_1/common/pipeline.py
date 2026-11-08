# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate stage-local V4.1 persistent and per-forward sharing dependencies."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SharingDependency:
    kind: str
    source_layer: int
    consumer_layer: int
    source_stage: int
    consumer_stage: int


def get_sharing_dependencies(
    config: Any, stage_ranges: list[tuple[int, int]]
) -> tuple[SharingDependency, ...]:
    """Resolve KV, index-K, top-k and candidate producers before construction."""
    num_layers = config.num_hidden_layers
    if (
        not stage_ranges
        or stage_ranges[0][0] != 0
        or stage_ranges[-1][1] != num_layers
        or any(start >= end for start, end in stage_ranges)
        or any(a[1] != b[0] for a, b in zip(stage_ranges, stage_ranges[1:]))
    ):
        raise ValueError("DeepSeek V4.1 pipeline stages must partition all layers")
    owners = {
        layer: stage
        for stage, (start, end) in enumerate(stage_ranges)
        for layer in range(start, end)
    }
    ratios = config.compress_ratios
    if len(ratios) < num_layers or any(r not in (0, 1, 2) for r in ratios):
        raise ValueError("V4.1 compress_ratios must cover layers with ratios 0/1/2")
    sources: dict[str, tuple[int, ...]] = {}
    for kind, field in (
        ("kv", "kv_source_layer_ids"),
        ("index", "index_source_layer_ids"),
    ):
        values = tuple(getattr(config, field, None) or ())
        if (
            any(
                type(layer) is not int or not 0 <= layer < num_layers
                for layer in values
            )
            or values != tuple(sorted(set(values)))
            or any(ratios[layer] == 0 for layer in values)
        ):
            raise ValueError(f"{field} must contain sorted, unique compressed layers")
        sources[kind] = values
    if not set(sources["kv"]).issubset(sources["index"]):
        raise ValueError("V4.1 KV sources must also publish index keys")
    dependencies = []

    def append(kind: str, source: int, consumer: int) -> None:
        if source != consumer:
            dependencies.append(
                SharingDependency(
                    kind, source, consumer, owners[source], owners[consumer]
                )
            )

    for layer in range(num_layers):
        if ratios[layer] == 0:
            continue
        for kind, values in sources.items():
            preceding = [source for source in values if source <= layer]
            if not preceding:
                raise ValueError(f"V4.1 layer {layer} has no preceding {kind} source")
            append(kind, preceding[-1], layer)
            if kind == "kv" and layer in sources["index"]:
                append("index_k", preceding[-1], layer)
    candidate = getattr(config, "candidate_source_layer_id", -1)
    if candidate >= 0 and getattr(config, "candidate_topk_blocks", 0) > 0:
        if candidate not in sources["index"]:
            raise ValueError("V4.1 candidate source must be an index source")
        for layer in sources["index"]:
            if layer > candidate:
                append("candidate", candidate, layer)
    return tuple(dependencies)


def validate_pipeline_partition(
    config: Any, stage_ranges: list[tuple[int, int]]
) -> None:
    """Validate the PP1/PP2 local and balanced PP4 transport contracts."""
    dependencies = get_sharing_dependencies(config, stage_ranges)
    cross_stage = [d for d in dependencies if d.source_stage != d.consumer_stage]
    pp4 = stage_ranges in (
        [(0, 10), (10, 20), (20, 30), (30, 40)],
        [(0, 10), (10, 20), (20, 31), (31, 40)],
        [(0, 10), (10, 20), (20, 32), (32, 40)],
    )
    if pp4:
        transported = {
            ("kv", 8, 0, 1),
            ("index", 8, 0, 1),
            ("kv", 20, 2, 3),
            ("index_k", 20, 2, 3),
            ("index", 28, 2, 3),
            ("candidate", 20, 2, 3),
        }
        cross_stage = [
            d
            for d in cross_stage
            if (d.kind, d.source_layer, d.source_stage, d.consumer_stage)
            not in transported
        ]
    if cross_stage:
        detail = "; ".join(
            f"{d.kind} source layer {d.source_layer} (stage {d.source_stage}) -> "
            f"layer {d.consumer_layer} (stage {d.consumer_stage})"
            for d in cross_stage[:4]
        )
        raise NotImplementedError(
            f"V4.1 pipeline crosses sharing dependencies: {detail}"
        )
    if len(stage_ranges) > 1 and not pp4 and stage_ranges != [(0, 20), (20, 40)]:
        raise NotImplementedError(
            "V4.1 PP supports only 20|20, 10|10|10|10, 10|10|11|9 "
            "or 10|10|12|8 partitions"
        )
    if len(stage_ranges) > 1 and any(
        layer >= 20 for layer in (getattr(config, "engram_layer_ids", None) or ())
    ):
        raise NotImplementedError("V4.1 PP requires Engram layers before layer 20")
