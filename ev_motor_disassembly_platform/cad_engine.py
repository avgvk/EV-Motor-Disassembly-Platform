from __future__ import annotations

import hashlib
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Any

import cadquery as cq
import numpy as np
import pandas as pd
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import (
    GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere,
    GeomAbs_Torus, GeomAbs_BSplineSurface, GeomAbs_BezierSurface,
    GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion,
)

SURFACE_NAMES = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
    GeomAbs_BSplineSurface: "bspline_surface",
    GeomAbs_BezierSurface: "bezier_surface",
    GeomAbs_SurfaceOfRevolution: "surface_of_revolution",
    GeomAbs_SurfaceOfExtrusion: "surface_of_extrusion",
}

ANALYTIC_AXIS_TYPES = {GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Torus, GeomAbs_SurfaceOfRevolution}


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])


def _canonical_axis(v: np.ndarray) -> np.ndarray:
    v = _unit(v)
    j = int(np.argmax(np.abs(v)))
    if v[j] < 0:
        v = -v
    return v


def _surface_axis(adaptor: BRepAdaptor_Surface, typ: Any) -> np.ndarray | None:
    try:
        if typ == GeomAbs_Cylinder:
            d = adaptor.Cylinder().Axis().Direction()
        elif typ == GeomAbs_Cone:
            d = adaptor.Cone().Axis().Direction()
        elif typ == GeomAbs_Torus:
            d = adaptor.Torus().Axis().Direction()
        elif typ == GeomAbs_SurfaceOfRevolution:
            d = adaptor.AxeOfRevolution().Direction()
        else:
            return None
        return _canonical_axis(np.array([d.X(), d.Y(), d.Z()], dtype=float))
    except Exception:
        return None


def _shape_bbox(shape: cq.Shape) -> dict[str, float]:
    b = shape.BoundingBox()
    return {
        "xmin": float(b.xmin), "ymin": float(b.ymin), "zmin": float(b.zmin),
        "xmax": float(b.xmax), "ymax": float(b.ymax), "zmax": float(b.zmax),
        "bbox_dx": float(b.xlen), "bbox_dy": float(b.ylen), "bbox_dz": float(b.zlen),
    }


def _mesh_shape(shape: cq.Shape, tolerance: float = 0.8, angular_tolerance: float = 0.15) -> dict[str, np.ndarray]:
    """Tessellate any CadQuery shape (solid or face) for Plotly/WebGL."""
    vertices, triangles = shape.tessellate(tolerance, angular_tolerance)
    pts = np.asarray([[float(v.x), float(v.y), float(v.z)] for v in vertices], dtype=np.float32)
    tri = np.asarray(triangles, dtype=np.int32)
    return {"points": pts, "triangles": tri}


def _mesh_solid(solid: cq.Shape, tolerance: float = 0.8, angular_tolerance: float = 0.15) -> dict[str, np.ndarray]:
    return _mesh_shape(solid, tolerance, angular_tolerance)


def load_step_bytes(data: bytes, filename: str = "uploaded.step", mesh_tolerance: float = 0.8) -> dict[str, Any]:
    suffix = os.path.splitext(filename)[1] or ".step"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        model = cq.importers.importStep(path)
        solids = model.solids().vals()
        if not solids:
            raise ValueError("No solids were found in the STEP file.")

        rows: list[dict[str, Any]] = []
        axis_samples: list[tuple[np.ndarray, float]] = []
        meshes: dict[int, dict[str, np.ndarray]] = {}

        for body_id, solid in enumerate(solids, start=1):
            center = solid.Center()
            bbox = _shape_bbox(solid)
            surface_counts: Counter[str] = Counter()
            face_count = 0
            axis_vectors: list[np.ndarray] = []
            analytic_radii: list[float] = []

            for local_face_id, face in enumerate(solid.Faces(), start=1):
                face_count += 1
                try:
                    adaptor = BRepAdaptor_Surface(face.wrapped, True)
                    typ = adaptor.GetType()
                    surface_counts[SURFACE_NAMES.get(typ, "other_surface")] += 1
                    area = float(face.Area())
                    axis = _surface_axis(adaptor, typ)
                    if axis is not None:
                        axis_vectors.append(axis)
                        axis_samples.append((axis, max(area, 1e-9)))
                    if typ == GeomAbs_Cylinder:
                        analytic_radii.append(float(adaptor.Cylinder().Radius()))
                    elif typ == GeomAbs_Sphere:
                        analytic_radii.append(float(adaptor.Sphere().Radius()))
                    elif typ == GeomAbs_Torus:
                        analytic_radii.extend([
                            float(adaptor.Torus().MajorRadius()),
                            float(adaptor.Torus().MinorRadius()),
                        ])
                except Exception:
                    surface_counts["other_surface"] += 1
            local_axis = None
            if axis_vectors:
                ref = axis_vectors[0]
                aligned = np.asarray([v if np.dot(v, ref) >= 0 else -v for v in axis_vectors])
                local_axis = _canonical_axis(aligned.mean(axis=0))

            row = {
                "body_id": body_id,
                "volume": float(solid.Volume()),
                "area": float(solid.Area()),
                "centroid_x": float(center.x), "centroid_y": float(center.y), "centroid_z": float(center.z),
                "num_faces": int(face_count),
                "num_edges": int(len(solid.Edges())),
                "surface_counts": dict(surface_counts),
                "sphere_faces": int(surface_counts.get("sphere", 0)),
                "cylinder_faces": int(surface_counts.get("cylinder", 0)),
                "plane_faces": int(surface_counts.get("plane", 0)),
                "torus_faces": int(surface_counts.get("torus", 0)),
                "mean_analytic_radius": float(np.mean(analytic_radii)) if analytic_radii else np.nan,
                "local_axis_x": float(local_axis[0]) if local_axis is not None else np.nan,
                "local_axis_y": float(local_axis[1]) if local_axis is not None else np.nan,
                "local_axis_z": float(local_axis[2]) if local_axis is not None else np.nan,
                **bbox,
            }
            rows.append(row)
            meshes[body_id] = _mesh_solid(solid, tolerance=mesh_tolerance)

        bodies = pd.DataFrame(rows)
        assembly_bbox = {
            "xmin": float(bodies.xmin.min()), "ymin": float(bodies.ymin.min()), "zmin": float(bodies.zmin.min()),
            "xmax": float(bodies.xmax.max()), "ymax": float(bodies.ymax.max()), "zmax": float(bodies.zmax.max()),
        }
        dims = np.array([
            assembly_bbox["xmax"] - assembly_bbox["xmin"],
            assembly_bbox["ymax"] - assembly_bbox["ymin"],
            assembly_bbox["zmax"] - assembly_bbox["zmin"],
        ])
        scale = float(max(dims.max(), 1e-9))
        assembly_center = np.array([
            (assembly_bbox["xmin"] + assembly_bbox["xmax"]) / 2,
            (assembly_bbox["ymin"] + assembly_bbox["ymax"]) / 2,
            (assembly_bbox["zmin"] + assembly_bbox["zmax"]) / 2,
        ])

        if axis_samples:
            seed = max(axis_samples, key=lambda x: x[1])[0]
            accum = np.zeros(3)
            for axis, weight in axis_samples:
                aligned = axis if np.dot(axis, seed) >= 0 else -axis
                accum += aligned * weight
            motor_axis = _canonical_axis(accum)
        else:
            # Fallback: longest assembly dimension.
            motor_axis = np.eye(3)[int(np.argmax(dims))]

        centroids = bodies[["centroid_x", "centroid_y", "centroid_z"]].to_numpy(float)
        rel = centroids - assembly_center
        axial = rel @ motor_axis
        radial_vec = rel - np.outer(axial, motor_axis)
        radial_offset = np.linalg.norm(radial_vec, axis=1)

        # Project each axis-aligned bbox's 8 corners onto inferred axis for robust extents.
        axial_lengths, radial_sizes = [], []
        for _, r in bodies.iterrows():
            corners = np.array([
                [x, y, z]
                for x in (r.xmin, r.xmax)
                for y in (r.ymin, r.ymax)
                for z in (r.zmin, r.zmax)
            ], dtype=float)
            c_rel = corners - assembly_center
            proj = c_rel @ motor_axis
            axial_lengths.append(float(proj.max() - proj.min()))
            radial = c_rel - np.outer(proj, motor_axis)
            # Diameter-like radial span around assembly axis.
            radial_sizes.append(float(np.linalg.norm(radial, axis=1).max()))

        bodies["volume_norm"] = bodies["volume"] / (scale ** 3)
        bodies["axial_position"] = axial
        bodies["axial_position_norm"] = axial / scale
        bodies["axis_offset"] = radial_offset
        bodies["axis_offset_norm"] = radial_offset / scale
        bodies["axial_length"] = axial_lengths
        bodies["axial_length_norm"] = bodies["axial_length"] / scale
        bodies["radial_size"] = radial_sizes
        bodies["radial_size_norm"] = bodies["radial_size"] / scale
        bodies["relative_motor_radius"] = bodies["radial_size_norm"]
        bodies["bbox_aspect"] = bodies[["bbox_dx", "bbox_dy", "bbox_dz"]].max(axis=1) / np.maximum(
            bodies[["bbox_dx", "bbox_dy", "bbox_dz"]].min(axis=1), 1e-9
        )

        digest = hashlib.sha256(data).hexdigest()
        return {
            "filename": filename,
            "sha256": digest,
            "model": model,
            "solids": solids,
            "bodies_df": bodies,
            "meshes": meshes,
            "motor_axis": motor_axis,
            "assembly_center": assembly_center,
            "assembly_scale": scale,
            "assembly_bbox": assembly_bbox,
        }
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def mesh_feature_face(project: dict[str, Any], body_id: int, local_face_id: int, tolerance: float = 0.35) -> dict[str, np.ndarray] | None:
    """Tessellate one extracted face on demand for interactive feature highlighting."""
    try:
        solid=project["solids"][int(body_id)-1]
        faces=solid.Faces()
        face=faces[int(local_face_id)-1]
        return _mesh_shape(face,tolerance=max(float(tolerance),0.08))
    except Exception:
        return None
