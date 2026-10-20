from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

SEMANTIC_CLASSES = [
    "rotor", "rotor_core", "stator", "shaft", "bearing", "motor_housing",
    "front_end_plate", "rear_end_plate", "end_plate_or_end_cap", "encoder",
    "sensor", "magnet", "fastener", "cooling_component", "electrical_connector",
    "repeated_internal_element", "other", "unknown",
]


def _geometry_signature(row: pd.Series) -> np.ndarray:
    dims = np.sort(np.array([row.bbox_dx, row.bbox_dy, row.bbox_dz], dtype=float))
    scale = max(float(dims[-1]), 1e-9)
    return np.array([
        row.volume / (scale ** 3),
        dims[0] / scale,
        dims[1] / scale,
        row.num_faces / 100.0,
        row.sphere_faces / max(row.num_faces, 1),
        row.cylinder_faces / max(row.num_faces, 1),
    ], dtype=float)


def _union_find(n: int):
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    return parent, find, union


def _cluster_1d(values: list[tuple[int, float]], threshold: float) -> list[list[int]]:
    if not values:
        return []
    values = sorted(values, key=lambda x: x[1])
    clusters = [[values[0][0]]]
    prev = values[0][1]
    for idx, val in values[1:]:
        if abs(val - prev) <= threshold:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
        prev = val
    return clusters


def automatic_grouping(bodies_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bodies = bodies_df.copy()
    n = len(bodies)
    signatures = [_geometry_signature(r) for _, r in bodies.iterrows()]
    parent, find, union = _union_find(n)

    # Repeated geometry families (rotation-insensitive bbox signature).
    for i in range(n):
        for j in range(i + 1, n):
            a, b = signatures[i], signatures[j]
            rel = np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-9)
            if rel < 0.04:
                union(i, j)

    fams: dict[int, list[int]] = {}
    for i in range(n):
        fams.setdefault(find(i), []).append(i)
    family_rows = []
    family_id_by_row = {}
    for fid, members in enumerate(sorted(fams.values(), key=lambda x: min(x)), start=1):
        for i in members:
            family_id_by_row[i] = fid
        family_rows.append({
            "repeat_family_id": fid,
            "body_ids": [int(bodies.iloc[i].body_id) for i in members],
            "family_size": len(members),
        })
    bodies["repeat_family_id"] = [family_id_by_row[i] for i in range(n)]
    family_sizes = {r["repeat_family_id"]: r["family_size"] for r in family_rows}
    bodies["repeat_family_size"] = bodies["repeat_family_id"].map(family_sizes)

    # Rolling-element candidates: sphere-dominant, compact, repeated.
    sphere_fraction = bodies.sphere_faces / np.maximum(bodies.num_faces, 1)
    rolling_mask = (
        (sphere_fraction >= 0.5)
        & (bodies.bbox_aspect <= 1.15)
        & (bodies.repeat_family_size >= 6)
    )
    rolling = bodies.loc[rolling_mask, ["body_id", "axial_position_norm"]]
    threshold = 0.035
    ball_clusters = _cluster_1d([(int(r.body_id), float(r.axial_position_norm)) for _, r in rolling.iterrows()], threshold)
    ball_clusters = [c for c in ball_clusters if len(c) >= 6]

    grouped: set[int] = set()
    group_rows: list[dict[str, Any]] = []
    gid = 1
    # Group rolling elements and attach up to two nearby coaxial ring-like bodies as probable races.
    for cluster in ball_clusters:
        cluster_rows = bodies[bodies.body_id.isin(cluster)]
        zc = float(cluster_rows.axial_position_norm.mean())
        orbit = float(cluster_rows.axis_offset_norm.mean())
        candidates = bodies[
            (~bodies.body_id.isin(cluster))
            & (bodies.axis_offset_norm < 0.04)
            & (np.abs(bodies.axial_position_norm - zc) < 0.06)
            & (bodies.cylinder_faces + bodies.torus_faces > 0)
        ].copy()
        if len(candidates):
            candidates["race_score"] = (
                np.abs(candidates.axial_position_norm - zc)
                + 0.5 * np.abs(candidates.radial_size_norm - orbit)
            )
            race_ids = candidates.sort_values("race_score").head(2).body_id.astype(int).tolist()
        else:
            race_ids = []
        ids = sorted(set(cluster + race_ids))
        grouped.update(ids)
        group_rows.append({"group_id": gid, "body_ids": ids, "group_kind": "bearing_candidate"})
        gid += 1

    # Remaining solids are candidate engineering components individually.
    for body_id in bodies.body_id.astype(int):
        if body_id not in grouped:
            group_rows.append({"group_id": gid, "body_ids": [body_id], "group_kind": "single_body"})
            gid += 1

    groups = pd.DataFrame(group_rows)
    agg_rows = []
    for _, g in groups.iterrows():
        ids = [int(x) for x in g.body_ids]
        part = bodies[bodies.body_id.isin(ids)]
        agg_rows.append({
            **g.to_dict(),
            "num_bodies": len(ids),
            "volume_norm": float(part.volume_norm.sum()),
            "relative_motor_radius": float(part.relative_motor_radius.max()),
            "axial_length_norm": float(part.axial_length_norm.max()),
            "axial_position_norm": float(np.average(part.axial_position_norm, weights=np.maximum(part.volume, 1e-12))),
            "axis_offset_norm": float(part.axis_offset_norm.min()),
            "num_faces": int(part.num_faces.sum()),
            "sphere_faces": int(part.sphere_faces.sum()),
            "cylinder_faces": int(part.cylinder_faces.sum()),
            "plane_faces": int(part.plane_faces.sum()),
            "repeat_family_size_max": int(part.repeat_family_size.max()),
        })
    groups = pd.DataFrame(agg_rows)

    body_to_group = {}
    for _, g in groups.iterrows():
        for bid in g.body_ids:
            body_to_group[int(bid)] = int(g.group_id)
    bodies["group_id"] = bodies.body_id.map(body_to_group)
    return bodies, groups


def heuristic_classification(groups_df: pd.DataFrame) -> pd.DataFrame:
    g = groups_df.copy()
    labels, confs, reasons = [], [], []
    max_radius = max(float(g.relative_motor_radius.max()), 1e-9)
    max_axial = max(float(np.abs(g.axial_position_norm).max()), 1e-9)

    # Identify outermost non-bearing group as housing prior.
    non_bearing = g[g.group_kind != "bearing_candidate"]
    housing_gid = int(non_bearing.sort_values("relative_motor_radius", ascending=False).iloc[0].group_id) if len(non_bearing) else -1

    for _, r in g.iterrows():
        radial_ratio = float(r.relative_motor_radius / max_radius)
        axial_ratio = float(abs(r.axial_position_norm) / max_axial) if max_axial else 0.0
        if r.group_kind == "bearing_candidate":
            label, conf, reason = "bearing", 0.93, "Repeated sphere-like bodies occur in one axial cluster with nearby coaxial ring candidates."
        elif int(r.group_id) == housing_gid:
            label, conf, reason = "motor_housing", 0.84, "This candidate reaches the largest radial envelope of the assembly."
        elif r.axis_offset_norm < 0.025 and r.axial_length_norm > 0.35 and radial_ratio < 0.22:
            label, conf, reason = "shaft", 0.80, "Long, narrow and nearly coaxial with the inferred motor axis."
        elif axial_ratio > 0.68 and r.axial_length_norm < 0.20 and radial_ratio > 0.45:
            label, conf, reason = "end_plate_or_end_cap", 0.74, "Broad radial extent, small axial thickness, and location near an axial end."
        elif r.axis_offset_norm < 0.05 and r.axial_length_norm > 0.18 and 0.45 <= radial_ratio < 0.86:
            label, conf, reason = "stator", 0.64, "Long coaxial annular candidate at an intermediate-to-large motor radius."
        elif r.axis_offset_norm < 0.05 and r.axial_length_norm > 0.18 and 0.20 <= radial_ratio < 0.55:
            label, conf, reason = "rotor", 0.64, "Long coaxial internal candidate at a smaller radius than the outer motor structure."
        elif axial_ratio > 0.60 and radial_ratio < 0.35:
            label, conf, reason = "encoder", 0.45, "Compact component near an axial end; encoder is a low-confidence engineering prior."
        elif r.repeat_family_size_max >= 4:
            label, conf, reason = "repeated_internal_element", 0.55, "This component belongs to a repeated geometry family."
        else:
            label, conf, reason = "unknown", 0.30, "Geometry alone does not provide enough evidence for a reliable semantic label."
        labels.append(label); confs.append(conf); reasons.append(reason)

    g["heuristic_label"] = labels
    g["heuristic_confidence"] = confs
    g["heuristic_reason"] = reasons
    g["api_label"] = None
    g["api_confidence"] = np.nan
    g["api_reason"] = None
    g["user_label"] = None
    g["final_component"] = g["heuristic_label"]
    g["final_confidence"] = g["heuristic_confidence"]
    g["label_source"] = "heuristic"
    return g


def build_relationships(groups: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for i, a in groups.iterrows():
        for j, b in groups.iterrows():
            if j <= i:
                continue
            rows.append({
                "group_id_1": int(a.group_id),
                "group_id_2": int(b.group_id),
                "axial_distance_norm": float(abs(a.axial_position_norm - b.axial_position_norm)),
                "radial_difference_norm": float(abs(a.relative_motor_radius - b.relative_motor_radius)),
                "both_coaxial": bool(a.axis_offset_norm < 0.05 and b.axis_offset_norm < 0.05),
            })
    return rows


def classify_with_openai(groups_df: pd.DataFrame, api_key: str, model: str = "gpt-5.6-terra") -> pd.DataFrame:
    from openai import OpenAI

    groups = groups_df.copy()
    payload_groups = []
    for _, r in groups.iterrows():
        payload_groups.append({
            "group_id": int(r.group_id),
            "num_bodies": int(r.num_bodies),
            "body_ids": [int(x) for x in r.body_ids],
            "volume_norm": float(r.volume_norm),
            "relative_motor_radius": float(r.relative_motor_radius),
            "axial_length_norm": float(r.axial_length_norm),
            "axial_position_norm": float(r.axial_position_norm),
            "axis_offset_norm": float(r.axis_offset_norm),
            "num_faces": int(r.num_faces),
            "sphere_faces": int(r.sphere_faces),
            "cylinder_faces": int(r.cylinder_faces),
            "plane_faces": int(r.plane_faces),
            "heuristic_label": str(r.heuristic_label),
            "heuristic_confidence": float(r.heuristic_confidence),
        })

    schema = {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "group_id": {"type": "integer"},
                        "predicted_component": {"type": "string", "enum": SEMANTIC_CLASSES},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                    },
                    "required": ["group_id", "predicted_component", "confidence", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["components"],
        "additionalProperties": False,
    }

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions=(
            "You classify candidate components from an EV motor STEP assembly. Original component names are unavailable. "
            "Use geometry, relative radius, axial position, coaxiality and repeated-body structure. Treat heuristic labels only as weak priors. "
            "Use unknown when evidence is weak. Rotor is normally internal and coaxial; stator surrounds rotor; housing is outer; shaft is long/narrow/coaxial; "
            "bearings often contain repeated rolling elements and races; end plates are broad and axially thin near motor ends."
        ),
        input=json.dumps({
            "assembly_type": "EV motor",
            "allowed_classes": SEMANTIC_CLASSES,
            "groups": payload_groups,
            "relationships": build_relationships(groups),
        }),
        text={"format": {"type": "json_schema", "name": "ev_motor_components", "strict": True, "schema": schema}},
    )
    result = json.loads(response.output_text)
    by_gid = {int(x["group_id"]): x for x in result["components"]}
    for idx, r in groups.iterrows():
        pred = by_gid.get(int(r.group_id))
        if pred:
            groups.at[idx, "api_label"] = pred["predicted_component"]
            groups.at[idx, "api_confidence"] = float(pred["confidence"])
            groups.at[idx, "api_reason"] = pred["reason"]
            if float(pred["confidence"]) >= 0.55:
                groups.at[idx, "final_component"] = pred["predicted_component"]
                groups.at[idx, "final_confidence"] = float(pred["confidence"])
                groups.at[idx, "label_source"] = "openai"
    return groups


def apply_user_labels(groups_df: pd.DataFrame) -> pd.DataFrame:
    groups = groups_df.copy()
    for idx, r in groups.iterrows():
        label = r.get("user_label")
        if isinstance(label, str) and label.strip() and label != "(keep automatic)":
            groups.at[idx, "final_component"] = label.strip()
            groups.at[idx, "final_confidence"] = 1.0
            groups.at[idx, "label_source"] = "user"
    return groups
