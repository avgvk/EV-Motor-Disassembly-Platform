from __future__ import annotations
import numpy as np
import pandas as pd

ROBOT_SYSTEM={
 "name":"Universal Robots UR10e + ATI QC-11 + Robotiq adaptive gripper",
 "robot":"UR10e","payload_kg":12.5,"reach_mm":1300,"repeatability_mm":0.05,"typical_power_w":350,"max_power_w":615,
 "tool_changer":"ATI QC-11","tool_changer_payload_kg":15.88,
 "gripper":"Robotiq adaptive 2-finger gripper","reference":"UR10e May 2026 datasheet; ATI QC-11 product page"
}
ASSESS_COLUMNS=["final_step","group_id","component","tool","estimated_disassembly_time_s","tool_change_time_s","disassembly_depth","stability_score","stability_risk","process_type","material_recovery_note","damage_possibility","estimated_energy_Wh","tooling_difficulty","tooling_difficulty_score","robot_feasibility","assessment_reason"]

def assess_disassembly(project,groups,sequence,relationships,tool_change_seconds=5.0,tcp_speed_m_s=0.20,fixture_clamped=True):
    if sequence is None or sequence.empty: return pd.DataFrame(columns=ASSESS_COLUMNS)
    scale_mm=float(project["assembly_scale"]); maxdepth=max(len(sequence),1); rows=[]; prev=None
    gmap=groups.set_index("group_id").to_dict("index")
    for _,r in sequence.sort_values("final_step").iterrows():
        gid=int(r.group_id); g=gmap.get(gid,{}); tool=str(r.tool); rel=relationships[(relationships.group_id_1==gid)|(relationships.group_id_2==gid)] if relationships is not None and not relationships.empty else pd.DataFrame()
        interfaces=" ".join(rel.final_interface_type.astype(str).tolist()) if not rel.empty else ""
        travel_m=max(0.08,min(0.35,(float(g.get("axial_length_norm",.1))*scale_mm+60)/1000.0)); motion=travel_m/max(tcp_speed_m_s,.03)
        fit_penalty=6 if ("bearing" in interfaces or "coaxial_fit" in interfaces) else 2
        setup=4; dis_t=setup+motion+fit_penalty; change=0 if prev in (None,tool) else float(tool_change_seconds); prev=tool
        depth=int(r.final_step)
        stability=max(.15,min(.98,(.92 if fixture_clamped else .68)-0.025*(depth-1)-(0.08 if str(r.component)=="motor_housing" else 0)))
        stability_risk="low" if stability>=.75 else ("medium" if stability>=.5 else "high")
        destructive="potentially destructive / damage-sensitive" if ("bearing" in interfaces or "fit_candidate" in interfaces) else "non-destructive preferred"
        damage="medium" if "bearing" in interfaces else ("low-medium" if "coaxial" in interfaces else "low")
        recovery="Material-specific recovery cannot be calculated because material metadata is absent; preserve part geometry and segregate after material identification."
        power=ROBOT_SYSTEM["typical_power_w"]; energy=power*(dis_t+change)/3600.0
        td=2 + (3 if "bearing" in interfaces else 0) + (2 if len(r.blocker_group_ids)>0 else 0) + (1 if float(r.component_confidence)<.6 else 0); td=min(10,td)
        difficulty="low" if td<=3 else ("medium" if td<=6 else "high")
        vol=float(g.get("volume_norm",0))*scale_mm**3; est_mass_kg=vol*2.7e-6 # aluminium-equivalent screening only
        feasible="screening pass" if est_mass_kg <= ROBOT_SYSTEM["payload_kg"]*.8 else "payload review required"
        rows.append({"final_step":depth,"group_id":gid,"component":str(r.component),"tool":tool,"estimated_disassembly_time_s":round(dis_t,2),"tool_change_time_s":round(change,2),"disassembly_depth":depth,"stability_score":round(stability,2),"stability_risk":stability_risk,"process_type":destructive,"material_recovery_note":recovery,"damage_possibility":damage,"estimated_energy_Wh":round(energy,3),"tooling_difficulty":difficulty,"tooling_difficulty_score":td,"robot_feasibility":feasible,"assessment_reason":"Parametric screening estimate from CAD geometry, sequence depth, interface type, selected robot profile and editable timing assumptions."})
    return pd.DataFrame(rows,columns=ASSESS_COLUMNS)

def improvement_suggestions(assessment,relationships,features):
    """Generate only evidence-linked, design-specific recommendations.

    Every row must point to a concrete component/step/relationship/feature in the
    uploaded design. Generic advice without an extracted trigger is intentionally
    omitted.
    """
    columns=[
        "recommendation_id","category","priority","group_id","component","final_step",
        "relationship_ids","feature_ids","current_issue","measured_evidence",
        "design_change","expected_disassembly_effect","circularity_benefit",
        "confidence","llm_suggestion","llm_reason","user_correction","final_recommendation"
    ]
    if assessment is None or assessment.empty:
        return pd.DataFrame(columns=columns)

    rel = relationships if relationships is not None else pd.DataFrame()
    feat = features if features is not None else pd.DataFrame()
    rows=[]

    def feature_ids_for_group(gid, limit=6):
        if feat is None or feat.empty or "group_id" not in feat.columns: return []
        q=feat[feat.group_id==gid].copy()
        if q.empty:return []
        if "area" in q.columns:q=q.sort_values("area",ascending=False)
        return q.head(limit).feature_id.astype(int).tolist()

    def add(category,priority,gid,component,step,rids,fids,issue,evidence,change,effect,benefit,confidence):
        rid=len(rows)+1
        rows.append({
            "recommendation_id":rid,"category":category,"priority":priority,"group_id":int(gid) if gid is not None else None,
            "component":str(component) if component is not None else "assembly","final_step":int(step) if step is not None else None,
            "relationship_ids":[int(x) for x in (rids or [])],"feature_ids":[int(x) for x in (fids or [])],
            "current_issue":issue,"measured_evidence":evidence,"design_change":change,
            "expected_disassembly_effect":effect,"circularity_benefit":benefit,"confidence":float(confidence),
            "llm_suggestion":None,"llm_reason":None,"user_correction":None,"final_recommendation":change,
        })

    # 1) High tooling difficulty: component-specific access/grasp redesign.
    for _,a in assessment[assessment.tooling_difficulty_score>=7].iterrows():
        gid=int(a.group_id)
        r=rel[(rel.group_id_1==gid)|(rel.group_id_2==gid)] if not rel.empty else pd.DataFrame()
        rids=r.relationship_id.astype(int).tolist()[:8] if not r.empty else []
        fids=feature_ids_for_group(gid)
        add(
            "repairability/remanufacturing","high",gid,a.component,a.final_step,rids,fids,
            "High robotic tooling difficulty for this specific removal step.",
            f"Tooling difficulty={int(a.tooling_difficulty_score)}/10; tool={a.tool}; damage possibility={a.damage_possibility}; mapped relationships={rids}.",
            f"Redesign Group {gid} ({a.component}) to provide an unobstructed robot grasp/tool-access region at mapped features {fids or 'not yet resolved'} and reduce the interface operations represented by relationships {rids or 'none mapped'}.",
            "Target fewer constrained approach motions and lower tooling difficulty for this identified step.",
            "Improves repair access and preserves the component for reuse/remanufacturing.",.86
        )

    # 2) Bearing/coaxial seat relationships: interface-specific separability recommendation.
    if not rel.empty:
        mask=rel.final_interface_type.astype(str).str.contains("bearing|coaxial",case=False,na=False) & (rel.distance_norm.astype(float) < 0.04) & (rel.confidence.astype(float) >= 0.60)
        for _,r in rel[mask].iterrows():
            # choose the more difficult/later of the two groups for traceability
            candidates=assessment[assessment.group_id.isin([int(r.group_id_1),int(r.group_id_2)])].sort_values("tooling_difficulty_score",ascending=False)
            a=candidates.iloc[0] if len(candidates) else None
            gid=int(a.group_id) if a is not None else int(r.group_id_1)
            comp=str(a.component) if a is not None else str(r.component_1)
            step=int(a.final_step) if a is not None else None
            fids=[]
            for col in ("mapped_feature_ids_1","mapped_feature_ids_2"):
                x=r.get(col,[])
                if isinstance(x,(list,tuple)): fids.extend(int(v) for v in x)
            fids=list(dict.fromkeys(fids))[:10]
            clearance=r.radial_clearance_estimate
            clearance_txt="unavailable" if pd.isna(clearance) else f"{float(clearance):.4g} STEP units"
            add(
                "remanufacturing/reuse","high",gid,comp,step,[int(r.relationship_id)],fids,
                f"Damage-sensitive {r.final_interface_type} between G{int(r.group_id_1)} and G{int(r.group_id_2)}.",
                f"Interface={r.final_interface_type}; mating={r.mating_condition}; coaxial={bool(r.both_coaxial)}; axial overlap={float(r.axial_overlap_ratio):.3f}; geometric radial-clearance proxy={clearance_txt}; tolerance={r.tolerance_status}.",
                f"Modify the G{int(r.group_id_1)}–G{int(r.group_id_2)} interface so removal is positively supported in the inferred extraction direction: add accessible puller/press lands or a serviceable retaining feature at mapped features {fids or 'to be localized'}, and specify a separable fit only after functional tolerance analysis.",
                "Target non-destructive extraction with less puller force and reduced seating-surface damage; numerical fit/tolerance must be validated outside geometry-only STEP evidence.",
                "Preserves bearing/shaft/housing seating surfaces for reuse and remanufacturing.",.90
            )

    # 3) Deep disassembly: identify the actual late components rather than generic modularity advice.
    threshold=max(6,int(np.ceil(len(assessment)*0.65)))
    for _,a in assessment[assessment.disassembly_depth>=threshold].iterrows():
        gid=int(a.group_id); fids=feature_ids_for_group(gid,4)
        r=rel[(rel.group_id_1==gid)|(rel.group_id_2==gid)] if not rel.empty else pd.DataFrame()
        rids=r.relationship_id.astype(int).tolist()[:8] if not r.empty else []
        add(
            "repairability/refurbishment","medium",gid,a.component,a.final_step,rids,fids,
            "This service candidate is reached late in the derived disassembly sequence.",
            f"Disassembly depth={int(a.disassembly_depth)} of {len(assessment)}; associated relationship IDs={rids}.",
            f"Repackage or expose Group {gid} ({a.component}) from the nearest motor end so relationships {rids or 'identified around the group'} do not require removal of as many preceding groups; preserve its functional interfaces {fids or 'identified faces'} while moving service access outward.",
            f"Target reducing access depth below {int(a.disassembly_depth)} steps; exact revised depth requires re-running the sequence after CAD modification.",
            "Improves repairability and refurbishment turnaround for this particular component.",.78
        )

    # 4) Stability issues: fixture/access redesign tied to specific step.
    for _,a in assessment[assessment.stability_risk.astype(str).isin(["medium","high"])].iterrows():
        gid=int(a.group_id); fids=feature_ids_for_group(gid,4)
        add(
            "repairability/robotic-disassembly","high" if a.stability_risk=="high" else "medium",gid,a.component,a.final_step,[],fids,
            f"Reduced assembly stability at disassembly step {int(a.final_step)}.",
            f"Screening stability score={float(a.stability_score):.2f} ({a.stability_risk}); component={a.component}.",
            f"Add or preserve a fixture datum/support surface that remains available through step {int(a.final_step)}; prioritize planar handling/locating features among {fids or 'the extracted faces'} and avoid using damage-sensitive mating faces as clamps.",
            "Target a higher support margin during this exact removal step; validate with fixture/contact simulation.",
            "Reduces accidental damage during repair/remanufacturing and improves robotic process robustness.",.74
        )

    # 5) Tool changes: name the exact transition pairs that drive the overhead.
    ordered=assessment.sort_values("final_step").reset_index(drop=True)
    for i in range(1,len(ordered)):
        prev,cur=ordered.iloc[i-1],ordered.iloc[i]
        if float(cur.tool_change_time_s)>0:
            add(
                "repairability/remanufacturing","medium",int(cur.group_id),cur.component,int(cur.final_step),[],feature_ids_for_group(int(cur.group_id),4),
                f"Automatic tool change between step {int(prev.final_step)} ({prev.tool}) and step {int(cur.final_step)} ({cur.tool}).",
                f"Tool-change penalty={float(cur.tool_change_time_s):.1f} s; transition G{int(prev.group_id)}→G{int(cur.group_id)}.",
                f"Where function permits, redesign the accessible interface/grasp geometry on G{int(cur.group_id)} so the operation can use `{prev.tool}` or reorder compatible interfaces without violating precedence.",
                f"Potentially removes this {float(cur.tool_change_time_s):.1f} s tool-change event; sequence must be recomputed to confirm.",
                "Reduces robotic disassembly time and process complexity for repair/remanufacturing.",.70
            )

    # 6) File-specific data gap affecting recovery: one assembly-level recommendation, explicitly evidence based.
    if assessment.material_recovery_note.astype(str).str.contains("material metadata is absent",case=False,na=False).any():
        add(
            "recycling/recovery","high",None,"assembly",None,[],[],
            "Material identity is absent from the imported geometry evidence, preventing material-specific recovery planning.",
            "Robotic assessment reports material metadata unavailable for the uploaded STEP; no alloy/polymer identity was extracted.",
            "Attach persistent material identifiers to the actual component groups (STEP PMI/properties, PLM linkage, or digital product passport) so each separable group can be routed to a verified recovery stream.",
            "Enables material-specific recovery decisions after disassembly; it does not change mechanical sequence depth by itself.",
            "Improves recycling/recovery traceability for this uploaded design.",.98
        )

    # Remove duplicates caused by the same interface being encountered through different triggers.
    out=pd.DataFrame(rows,columns=columns)
    if out.empty:return out
    out=out.drop_duplicates(subset=["category","group_id","current_issue","design_change"]).reset_index(drop=True)
    out["recommendation_id"]=range(1,len(out)+1)
    return out[columns]

