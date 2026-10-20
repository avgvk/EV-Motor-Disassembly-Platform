from __future__ import annotations
import json
import pandas as pd
import numpy as np

def _client(key):
    from openai import OpenAI
    return OpenAI(api_key=key)

def reason_component_features(df,api_key,model):
    if df is None or df.empty: return df
    sample=df.sort_values("area",ascending=False).head(160)
    payload=[{"feature_id":int(r.feature_id),"feature_type":str(r.feature_type),"surface_type":str(r.surface_type),"dimensions":[None if pd.isna(r.dimension_1) else float(r.dimension_1),None if pd.isna(r.dimension_2) else float(r.dimension_2)],"position":[float(r.position_x),float(r.position_y),float(r.position_z)],"geometry_function":str(r.function_class)} for _,r in sample.iterrows()]
    schema={"type":"object","properties":{"features":{"type":"array","items":{"type":"object","properties":{"feature_id":{"type":"integer"},"function_class":{"type":"string"},"confidence":{"type":"number"},"reason":{"type":"string"}},"required":["feature_id","function_class","confidence","reason"],"additionalProperties":False}}},"required":["features"],"additionalProperties":False}
    resp=_client(api_key).responses.create(model=model,instructions="Classify CAD features for robotic disassembly. Use only supplied geometry. Allowed meanings include handling, inserting, fastening, locating, mating, rolling, unknown. Do not invent holes or dimensions.",input=json.dumps(payload),text={"format":{"type":"json_schema","name":"feature_reasoning","strict":True,"schema":schema}})
    out=df.copy(); res=json.loads(resp.output_text); by={int(x["feature_id"]):x for x in res["features"]}
    for i,r in out.iterrows():
        x=by.get(int(r.feature_id));
        if x: out.at[i,"llm_function_class"]=x["function_class"]; out.at[i,"llm_reason"]=x["reason"]
    return out

def reason_relationships(df,api_key,model):
    if df is None or df.empty: return df
    sample=df.sort_values("distance_norm").head(120)
    payload=[{"relationship_id":int(r.relationship_id),"components":[str(r.component_1),str(r.component_2)],"distance_norm":float(r.distance_norm),"geometry_interface":str(r.interface_type),"coaxial":bool(r.both_coaxial),"axial_overlap":float(r.axial_overlap_ratio),"clearance_estimate":None if pd.isna(r.radial_clearance_estimate) else float(r.radial_clearance_estimate),"tolerance_status":str(r.tolerance_status)} for _,r in sample.iterrows()]
    schema={"type":"object","properties":{"relationships":{"type":"array","items":{"type":"object","properties":{"relationship_id":{"type":"integer"},"interface_type":{"type":"string"},"likely_fit":{"type":"string"},"precedence":{"type":"string"},"confidence":{"type":"number"},"reason":{"type":"string"}},"required":["relationship_id","interface_type","likely_fit","precedence","confidence","reason"],"additionalProperties":False}}},"required":["relationships"],"additionalProperties":False}
    resp=_client(api_key).responses.create(model=model,instructions="Reason about EV-motor component relationships using only supplied CAD evidence. Never fabricate numeric tolerances. If GD&T/PMI is missing, say fit is only a likely engineering interpretation. Precedence must be evidence-based or 'undetermined'.",input=json.dumps(payload),text={"format":{"type":"json_schema","name":"relationship_reasoning","strict":True,"schema":schema}})
    out=df.copy(); res=json.loads(resp.output_text); by={int(x["relationship_id"]):x for x in res["relationships"]}
    for i,r in out.iterrows():
        x=by.get(int(r.relationship_id));
        if x: out.at[i,"llm_interface_type"]=x["interface_type"]; out.at[i,"llm_fit"]=x["likely_fit"]; out.at[i,"llm_precedence"]=x["precedence"]; out.at[i,"llm_reason"]=x["reason"]
    return out

def reason_sequence(seq,relationships,api_key,model):
    if seq is None or seq.empty: return seq
    payload=[{"group_id":int(r.group_id),"geometry_step":int(r.step),"component":str(r.component),"direction":str(r.removal_direction),"blockers":[int(x) for x in r.blocker_group_ids],"interfaces":[str(x) for x in r.interface_types],"reason":str(r.geometry_reason)} for _,r in seq.iterrows()]
    schema={"type":"object","properties":{"sequence":{"type":"array","items":{"type":"object","properties":{"group_id":{"type":"integer"},"step":{"type":"integer"},"direction":{"type":"string","enum":["+motor_axis","-motor_axis"]},"confidence":{"type":"number"},"reason":{"type":"string"}},"required":["group_id","step","direction","confidence","reason"],"additionalProperties":False}}},"required":["sequence"],"additionalProperties":False}
    resp=_client(api_key).responses.create(model=model,instructions="Review a geometry-derived EV motor disassembly sequence. Do not invent fasteners or contacts. Respect blockers. If evidence is weak, preserve geometry order. Return each group once.",input=json.dumps(payload),text={"format":{"type":"json_schema","name":"sequence_reasoning","strict":True,"schema":schema}})
    out=seq.copy(); res=json.loads(resp.output_text); by={int(x["group_id"]):x for x in res["sequence"]}
    for i,r in out.iterrows():
        x=by.get(int(r.group_id));
        if x: out.at[i,"llm_step"]=int(x["step"]); out.at[i,"llm_direction"]=x["direction"]; out.at[i,"llm_confidence"]=float(x["confidence"]); out.at[i,"llm_reason"]=x["reason"]
    return out

def reason_assessment(assessment,api_key,model):
    if assessment is None or assessment.empty: return assessment
    payload=assessment.to_dict("records")
    schema={"type":"object","properties":{"steps":{"type":"array","items":{"type":"object","properties":{"group_id":{"type":"integer"},"feasibility":{"type":"string"},"risk":{"type":"string"},"reason":{"type":"string"}},"required":["group_id","feasibility","risk","reason"],"additionalProperties":False}}},"required":["steps"],"additionalProperties":False}
    resp=_client(api_key).responses.create(model=model,instructions="Review a robotic EV-motor disassembly screening assessment. Treat numeric timing/energy values as model outputs, not measured facts. Identify feasibility/risk using only supplied step data. Do not invent robot capabilities or material properties.",input=json.dumps(payload,default=str),text={"format":{"type":"json_schema","name":"robot_assessment_reasoning","strict":True,"schema":schema}})
    out=assessment.copy();
    for c in ["llm_feasibility","llm_risk","llm_reason","user_assessment","final_assessment"]:
        if c not in out.columns: out[c]=None
    res=json.loads(resp.output_text); by={int(x["group_id"]):x for x in res["steps"]}
    for i,r in out.iterrows():
        x=by.get(int(r.group_id))
        if x:
            out.at[i,"llm_feasibility"]=x["feasibility"]; out.at[i,"llm_risk"]=x["risk"]; out.at[i,"llm_reason"]=x["reason"]
            out.at[i,"final_assessment"]=f"{x['feasibility']} | {x['risk']}"
    return out

def reason_improvements(suggestions,assessment,relationships,features,api_key,model):
    if suggestions is None or suggestions.empty:
        return suggestions
    payload={
        "geometry_recommendations":suggestions.to_dict("records"),
        "assessment":assessment.to_dict("records") if assessment is not None else [],
        "relationships":relationships.to_dict("records") if relationships is not None else [],
        "features":features.to_dict("records") if features is not None else [],
    }
    schema={"type":"object","properties":{"recommendations":{"type":"array","items":{"type":"object","properties":{
        "recommendation_id":{"type":"integer"},"suggestion":{"type":"string"},"reason":{"type":"string"},"confidence":{"type":"number"}
    },"required":["recommendation_id","suggestion","reason","confidence"],"additionalProperties":False}}},"required":["recommendations"],"additionalProperties":False}
    instructions=(
        "Review design-specific EV-motor disassembly recommendations. Every recommendation MUST remain tied to the supplied recommendation_id and its exact "
        "group_id/component, feature_ids, relationship_ids, measured_evidence and disassembly step. Do not output generic design-for-disassembly advice. "
        "Do not invent holes, fasteners, materials, fits, tolerances, forces, or accessibility that are not in the evidence. If the evidence cannot support a "
        "specific modification, return 'Insufficient design evidence' as the suggestion and explain what data is missing. Numerical tolerance classes must never "
        "be fabricated from a geometry-only STEP file."
    )
    resp=_client(api_key).responses.create(model=model,instructions=instructions,input=json.dumps(payload,default=str),text={"format":{"type":"json_schema","name":"design_specific_improvements","strict":True,"schema":schema}})
    out=suggestions.copy()
    res=json.loads(resp.output_text); by={int(x["recommendation_id"]):x for x in res["recommendations"]}
    for i,r in out.iterrows():
        x=by.get(int(r.recommendation_id))
        if x:
            out.at[i,"llm_suggestion"]=x["suggestion"]
            out.at[i,"llm_reason"]=x["reason"]
            # Keep deterministic CAD recommendation as final until user explicitly accepts/corrects.
            if not isinstance(out.at[i,"final_recommendation"],str) or not out.at[i,"final_recommendation"]:
                out.at[i,"final_recommendation"]=r.design_change
    return out

