from __future__ import annotations

from typing import Any
import math
import numpy as np
import pandas as pd
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus

COMPONENT_FEATURE_COLUMNS = [
    "feature_id","group_id","body_id","local_face_id","feature_type","surface_type","function_class",
    "dimension_1","dimension_2","dimension_3","position_x","position_y","position_z",
    "orientation_x","orientation_y","orientation_z","area","confidence","evidence","llm_function_class",
    "llm_reason","user_function_class","final_function_class"
]

RELATION_COLUMNS = [
    "relationship_id","group_id_1","group_id_2","component_1","component_2","distance","distance_norm",
    "spatial_adjacency","topology_relation","interface_type","mating_condition","both_coaxial","axial_overlap_ratio",
    "radial_clearance_estimate","fit_geometry","tolerance_status","precedence_hint","mapped_feature_ids_1",
    "mapped_feature_ids_2","confidence","evidence","llm_interface_type","llm_fit","llm_precedence","llm_reason",
    "user_interface_type","user_fit","user_precedence","final_interface_type","final_fit","final_precedence"
]


def _unit(v):
    a=np.asarray(v,dtype=float); n=float(np.linalg.norm(a)); return a/n if n>1e-12 else np.array([1.,0.,0.])


def _face_normal(face):
    try:
        n=face.normalAt()
        return _unit([n.x,n.y,n.z])
    except Exception:
        return np.array([np.nan,np.nan,np.nan])


def _face_axis(adaptor, typ):
    try:
        if typ == GeomAbs_Cylinder: d=adaptor.Cylinder().Axis().Direction()
        elif typ == GeomAbs_Cone: d=adaptor.Cone().Axis().Direction()
        elif typ == GeomAbs_Torus: d=adaptor.Torus().Axis().Direction()
        else: return None
        return _unit([d.X(),d.Y(),d.Z()])
    except Exception:
        return None


def extract_component_features(project: dict[str,Any], groups_df: pd.DataFrame) -> pd.DataFrame:
    solids=project["solids"]
    body_to_group={}
    for _,g in groups_df.iterrows():
        for bid in g.body_ids: body_to_group[int(bid)] = int(g.group_id)
    rows=[]; fid=1
    for body_id, solid in enumerate(solids,start=1):
        gid=body_to_group.get(body_id)
        if gid is None: continue
        body_scale=max(float(solid.BoundingBox().DiagonalLength),1e-9)
        for local_face_id,face in enumerate(solid.Faces(),start=1):
            try:
                ad=BRepAdaptor_Surface(face.wrapped,True); typ=ad.GetType()
                c=face.Center(); area=float(face.Area()); normal=_face_normal(face)
                bb=face.BoundingBox(); dims=sorted([float(bb.xlen),float(bb.ylen),float(bb.zlen)],reverse=True)
                surface_type="other"; feature_type="surface"; function="handling"; d1=d2=d3=np.nan; orient=normal
                evidence=[]; conf=.45
                if typ==GeomAbs_Plane:
                    surface_type="plane"; feature_type="planar_face"; function="mating_or_handling"; d1,d2,d3=dims; conf=.72
                    evidence.append("planar face measured from BRep")
                    # elongated planar pocket/slot proxy
                    if dims[0] > 2.8*max(dims[1],1e-9) and area < 0.08*body_scale*body_scale:
                        feature_type="slot_candidate"; function="locating_or_fastening"; conf=.52; evidence.append("elongated planar face proxy")
                elif typ==GeomAbs_Cylinder:
                    surface_type="cylinder"; r=float(ad.Cylinder().Radius()); orient=_face_axis(ad,typ); d1=2*r; d2=dims[0]; d3=np.nan
                    circ=sum(1 for e in face.Edges() if str(e.geomType()).upper()=="CIRCLE")
                    if circ>=1 and r < 0.18*body_scale:
                        feature_type="hole_or_bore_candidate"; function="fastening_or_inserting"; conf=.78; evidence.append(f"cylindrical face with {circ} circular boundary edge(s)")
                    else:
                        feature_type="cylindrical_mating_surface"; function="mating_inserting_or_handling"; conf=.76; evidence.append("cylindrical BRep surface")
                elif typ==GeomAbs_Cone:
                    surface_type="cone"; orient=_face_axis(ad,typ); feature_type="conical_mating_surface"; function="locating_or_inserting"; d1=dims[0]; d2=dims[1]; conf=.70
                elif typ==GeomAbs_Sphere:
                    surface_type="sphere"; feature_type="spherical_surface"; function="rolling_or_handling"; d1=2*float(ad.Sphere().Radius()); conf=.80
                elif typ==GeomAbs_Torus:
                    surface_type="torus"; feature_type="toroidal_surface"; function="rolling_or_mating"; orient=_face_axis(ad,typ); d1=2*float(ad.Torus().MajorRadius()); d2=2*float(ad.Torus().MinorRadius()); conf=.72
                else:
                    surface_type=str(typ); feature_type="freeform_surface"; function="handling_or_mating"; d1,d2,d3=dims
                rows.append({
                    "feature_id":fid,"group_id":gid,"body_id":body_id,"local_face_id":local_face_id,
                    "feature_type":feature_type,"surface_type":surface_type,"function_class":function,
                    "dimension_1":float(d1) if np.isfinite(d1) else np.nan,"dimension_2":float(d2) if np.isfinite(d2) else np.nan,
                    "dimension_3":float(d3) if np.isfinite(d3) else np.nan,"position_x":float(c.x),"position_y":float(c.y),"position_z":float(c.z),
                    "orientation_x":float(orient[0]) if np.isfinite(orient[0]) else np.nan,"orientation_y":float(orient[1]) if np.isfinite(orient[1]) else np.nan,
                    "orientation_z":float(orient[2]) if np.isfinite(orient[2]) else np.nan,"area":area,"confidence":conf,
                    "evidence":"; ".join(evidence),"llm_function_class":None,"llm_reason":None,"user_function_class":None,
                    "final_function_class":function,
                }); fid += 1
            except Exception:
                continue
    return pd.DataFrame(rows, columns=COMPONENT_FEATURE_COLUMNS)


def _bbox_gap(a,b):
    gaps=[]
    for lo1,hi1,lo2,hi2 in [(a.xmin,a.xmax,b.xmin,b.xmax),(a.ymin,a.ymax,b.ymin,b.ymax),(a.zmin,a.zmax,b.zmin,b.zmax)]:
        gaps.append(max(0.0,float(lo2-hi1),float(lo1-hi2)))
    return float(np.linalg.norm(gaps))


def _group_geom(groups,bodies):
    out=[]
    for _,g in groups.iterrows():
        p=bodies[bodies.body_id.isin([int(x) for x in g.body_ids])]
        out.append({"group_id":int(g.group_id),"xmin":p.xmin.min(),"xmax":p.xmax.max(),"ymin":p.ymin.min(),"ymax":p.ymax.max(),"zmin":p.zmin.min(),"zmax":p.zmax.max(),
                    "axial_position_norm":float(g.axial_position_norm),"axial_length_norm":float(g.axial_length_norm),"relative_motor_radius":float(g.relative_motor_radius),"axis_offset_norm":float(g.axis_offset_norm)})
    return pd.DataFrame(out)


def extract_relationships(project: dict[str,Any], groups_df: pd.DataFrame, component_features_df: pd.DataFrame) -> pd.DataFrame:
    scale=float(project["assembly_scale"]); geom=_group_geom(groups_df,project["bodies_df"]); labels=groups_df.set_index("group_id")["final_component"].to_dict()
    rows=[]; rid=1
    for i,a in geom.iterrows():
        for j,b in geom.iterrows():
            if j<=i: continue
            gap=_bbox_gap(a,b); gapn=gap/max(scale,1e-9)
            a0,a1=float(a.axial_position_norm-a.axial_length_norm/2),float(a.axial_position_norm+a.axial_length_norm/2)
            b0,b1=float(b.axial_position_norm-b.axial_length_norm/2),float(b.axial_position_norm+b.axial_length_norm/2)
            overlap=max(0.0,min(a1,b1)-max(a0,b0)); denom=max(min(float(a.axial_length_norm),float(b.axial_length_norm)),1e-9); overlap_ratio=min(1.,overlap/denom)
            coax=bool(a.axis_offset_norm<0.05 and b.axis_offset_norm<0.05)
            clearance=abs(float(a.relative_motor_radius-b.relative_motor_radius))*scale if coax else np.nan
            adjacency="contact_or_near" if gapn<0.01 else ("near" if gapn<0.04 else "separated")
            nested=coax and overlap_ratio>0.25
            if nested and {labels.get(int(a.group_id)),labels.get(int(b.group_id))} & {"bearing"}:
                interface="bearing_seat_candidate"; mating="coaxial radial seat"
            elif nested:
                interface="cylindrical_or_coaxial_fit_candidate"; mating="coaxial insertion/nesting"
            elif gapn<0.01 and overlap_ratio<0.25:
                interface="end_or_planar_interface_candidate"; mating="planar/end contact"
            elif gapn<0.04:
                interface="near_interface"; mating="near-contact/proximity"
            else:
                interface="no_direct_interface"; mating="none detected"
            fitgeom="clearance_or_transition_candidate" if nested and clearance>0 else ("contact_candidate" if gapn<0.01 else "unknown")
            tol="not available from geometric STEP evidence; PMI/GD&T required"
            prec="outer_or_end_first" if abs(float(a.axial_position_norm))!=abs(float(b.axial_position_norm)) else "undetermined"
            fa=component_features_df[component_features_df.group_id==int(a.group_id)].nlargest(4,"area").feature_id.astype(int).tolist() if not component_features_df.empty else []
            fb=component_features_df[component_features_df.group_id==int(b.group_id)].nlargest(4,"area").feature_id.astype(int).tolist() if not component_features_df.empty else []
            conf=.82 if gapn<.01 else (.68 if gapn<.04 else .35)
            rows.append({"relationship_id":rid,"group_id_1":int(a.group_id),"group_id_2":int(b.group_id),"component_1":labels.get(int(a.group_id),"unknown"),"component_2":labels.get(int(b.group_id),"unknown"),
                         "distance":gap,"distance_norm":gapn,"spatial_adjacency":adjacency,"topology_relation":"nested/coaxial" if nested else "separate bodies",
                         "interface_type":interface,"mating_condition":mating,"both_coaxial":coax,"axial_overlap_ratio":overlap_ratio,"radial_clearance_estimate":clearance,
                         "fit_geometry":fitgeom,"tolerance_status":tol,"precedence_hint":prec,"mapped_feature_ids_1":fa,"mapped_feature_ids_2":fb,"confidence":conf,
                         "evidence":"BRep bounding envelopes + motor-axis geometry + component features","llm_interface_type":None,"llm_fit":None,"llm_precedence":None,"llm_reason":None,
                         "user_interface_type":None,"user_fit":None,"user_precedence":None,"final_interface_type":interface,"final_fit":fitgeom,"final_precedence":prec}); rid+=1
    return pd.DataFrame(rows,columns=RELATION_COLUMNS)


def relationship_matrix(groups_df: pd.DataFrame, relationships_df: pd.DataFrame) -> pd.DataFrame:
    gids=[int(x) for x in groups_df.group_id]
    labels={int(r.group_id):str(r.final_component) for _,r in groups_df.iterrows()}
    names=[f"G{g}:{labels[g]}" for g in gids]
    m=pd.DataFrame("—",index=names,columns=names)
    for n in names: m.loc[n,n]="SELF"
    for _,r in relationships_df.iterrows():
        a=f"G{int(r.group_id_1)}:{labels.get(int(r.group_id_1),'?')}"; b=f"G{int(r.group_id_2)}:{labels.get(int(r.group_id_2),'?')}"
        val=f"{r.final_interface_type} | {r.final_fit} | {r.final_precedence}"
        m.loc[a,b]=val; m.loc[b,a]=val
    return m
