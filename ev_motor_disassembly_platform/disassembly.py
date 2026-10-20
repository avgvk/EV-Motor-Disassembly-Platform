from __future__ import annotations
from typing import Any
import json
import numpy as np
import pandas as pd

SEQUENCE_COLUMNS=["step","group_id","component","component_confidence","removal_direction","direction_x","direction_y","direction_z","blocker_group_ids","relationship_ids","assembly_feature_ids","interface_types","tool","sequence_confidence","needs_reasoning","geometry_reason","llm_step","llm_direction","llm_confidence","llm_reason","user_step","user_direction","final_step","final_direction","source","reason"]


def _empty(): return pd.DataFrame(columns=SEQUENCE_COLUMNS)

def _group_geom(project,groups):
    bodies=project["bodies_df"]; out=[]
    for _,g in groups.iterrows():
        p=bodies[bodies.body_id.isin([int(x) for x in g.body_ids])]
        out.append({"group_id":int(g.group_id),"axis_center":float(g.axial_position_norm),"axial_length":float(g.axial_length_norm),"radial":float(g.relative_motor_radius),"label":str(g.final_component),"conf":float(g.final_confidence),"volume":float(p.volume.sum())})
    return pd.DataFrame(out)

def _tool(label, interface_types):
    s=" ".join(interface_types)
    if "bearing" in s or label=="bearing": return "bearing puller + gripper"
    if "fasten" in s or label=="fastener": return "electric screwdriver"
    if label in {"motor_housing","end_plate_or_end_cap","front_end_plate","rear_end_plate"}: return "parallel gripper"
    return "adaptive gripper"

def generate_geometry_sequence(project:dict[str,Any],groups_df:pd.DataFrame,relationships_df:pd.DataFrame,features_df:pd.DataFrame|None=None)->pd.DataFrame:
    if groups_df is None or groups_df.empty: return _empty()
    geom=_group_geom(project,groups_df); axis=np.asarray(project["motor_axis"],float); remaining=geom.copy(); rows=[]; step=1
    maxpos=max(float(np.abs(geom.axis_center).max()),1e-9)
    while not remaining.empty:
        cand=[]
        for _,t in remaining.iterrows():
            others=remaining[remaining.group_id!=int(t.group_id)]
            plus=others[(others.axis_center>t.axis_center+0.01)&(others.radial>0.35*t.radial)].group_id.astype(int).tolist()
            minus=others[(others.axis_center<t.axis_center-0.01)&(others.radial>0.35*t.radial)].group_id.astype(int).tolist()
            if len(plus)<len(minus): sign=1; blockers=plus
            elif len(minus)<len(plus): sign=-1; blockers=minus
            else: sign=1 if t.axis_center>=0 else -1; blockers=plus if sign>0 else minus
            rel=relationships_df[((relationships_df.group_id_1==int(t.group_id))|(relationships_df.group_id_2==int(t.group_id))) & (relationships_df.distance_norm<.04)] if relationships_df is not None and not relationships_df.empty else pd.DataFrame()
            rid=rel.relationship_id.astype(int).tolist() if not rel.empty else []
            interfaces=sorted(rel.final_interface_type.astype(str).unique().tolist()) if not rel.empty else []
            fids=[]
            if features_df is not None and not features_df.empty: fids=features_df[features_df.group_id==int(t.group_id)].nlargest(6,"area").feature_id.astype(int).tolist()
            exposure=abs(float(t.axis_center))/maxpos; score=100*len(blockers)+2*len(rid)-4*exposure
            cand.append((score,t,sign,blockers,rid,interfaces,fids,exposure))
        cand.sort(key=lambda x:x[0]); _,t,sign,blockers,rids,interfaces,fids,exposure=cand[0]; d=axis*sign
        amb=bool(blockers) or float(t.conf)<.6
        reason=f"geometry: {len(blockers)} axial blocker(s), {len(rids)} near/interface relation(s), end exposure {exposure:.2f}"
        rows.append({"step":step,"group_id":int(t.group_id),"component":str(t.label),"component_confidence":float(t.conf),"removal_direction":"+motor_axis" if sign>0 else "-motor_axis","direction_x":float(d[0]),"direction_y":float(d[1]),"direction_z":float(d[2]),"blocker_group_ids":blockers,"relationship_ids":rids,"assembly_feature_ids":fids,"interface_types":interfaces,"tool":_tool(str(t.label),interfaces),"sequence_confidence":max(.2,min(.95,.9-.12*len(blockers)-(.15 if amb else 0))),"needs_reasoning":amb,"geometry_reason":reason,"llm_step":np.nan,"llm_direction":None,"llm_confidence":np.nan,"llm_reason":None,"user_step":np.nan,"user_direction":None,"final_step":step,"final_direction":"+motor_axis" if sign>0 else "-motor_axis","source":"geometry","reason":reason})
        remaining=remaining[remaining.group_id!=int(t.group_id)].copy(); step+=1
    return pd.DataFrame(rows,columns=SEQUENCE_COLUMNS)


def apply_sequence_corrections(df:pd.DataFrame)->pd.DataFrame:
    if df is None or df.empty: return _empty()
    out=df.copy()
    out["final_step"]=out.apply(lambda r:int(r.user_step) if pd.notna(r.user_step) else (int(r.llm_step) if pd.notna(r.llm_step) else int(r.step)),axis=1)
    out["final_direction"]=out.apply(lambda r:r.user_direction if isinstance(r.user_direction,str) and r.user_direction else (r.llm_direction if isinstance(r.llm_direction,str) and r.llm_direction else r.removal_direction),axis=1)
    out["source"]=out.apply(lambda r:"user" if pd.notna(r.user_step) or (isinstance(r.user_direction,str) and r.user_direction) else ("llm" if pd.notna(r.llm_step) else "geometry"),axis=1)
    return out.sort_values(["final_step","group_id"]).reset_index(drop=True)
