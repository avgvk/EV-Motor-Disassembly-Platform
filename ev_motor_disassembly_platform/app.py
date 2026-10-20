from __future__ import annotations
import io, json, os
import pandas as pd
import numpy as np
import streamlit as st

from cad_engine import load_step_bytes, mesh_feature_face
from classifier import SEMANTIC_CLASSES, automatic_grouping, heuristic_classification, classify_with_openai, apply_user_labels
from feature_engine import extract_component_features, extract_relationships, relationship_matrix
from disassembly import generate_geometry_sequence, apply_sequence_corrections, SEQUENCE_COLUMNS
from assessment import ROBOT_SYSTEM, assess_disassembly, improvement_suggestions
from llm_reasoner import reason_component_features, reason_relationships, reason_sequence, reason_assessment, reason_improvements
from viewer import make_figure


def _server_setting(name, default=None):
    """Read server-side Streamlit secrets first, then environment variables."""
    try:
        value=st.secrets[name]
        if value is not None and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    value=os.getenv(name,default)
    return value.strip() if isinstance(value,str) else value


st.set_page_config(page_title="EV Motor Disassembly Intelligence",page_icon="⚙️",layout="wide")
st.title("EV Motor Disassembly Intelligence Platform")
st.caption("STEP → visualization → component identification → assembly features → relationship matrix → sequence → circularity evaluation → robotic assessment → circularity/design improvements")

STATE={"project":None,"groups":None,"bodies":None,"features":None,"relationships":None,"sequence":None,"assessment":None,"suggestions":None,"circularity_evaluation":None}
for k,v in STATE.items():
    if k not in st.session_state: st.session_state[k]=v

with st.sidebar:
    st.header("Project")
    uploaded=st.file_uploader("Upload STEP/STP",type=["step","stp"])
    mesh_tol=st.slider("Display mesh tolerance",0.2,2.0,0.8,0.1)
    process=st.button("Process STEP",type="primary",use_container_width=True,disabled=uploaded is None)
    st.divider()
    st.header("LLM reasoning")
    api_key=_server_setting("OPENAI_API_KEY","")
    model_name=_server_setting("OPENAI_MODEL","gpt-5.6-terra")
    if api_key:
        st.success("LLM key configured on the server")
    else:
        st.info("LLM key not configured. Add OPENAI_API_KEY to .streamlit/secrets.toml or the server environment.")
    st.caption(f"Model: {model_name}. The API key is never displayed or entered through the browser UI.")
    st.caption("Every LLM result is stored beside the geometry result; user corrections remain the final authority.")

if process and uploaded:
    try:
        with st.spinner("Reading STEP and tessellating solids..."):
            p=load_step_bytes(uploaded.getvalue(),uploaded.name,mesh_tolerance=mesh_tol)
            bodies,groups=automatic_grouping(p["bodies_df"]); groups=heuristic_classification(groups); p["bodies_df"]=bodies
            st.session_state.update(project=p,bodies=bodies,groups=groups,features=None,relationships=None,sequence=None,assessment=None,suggestions=None,circularity_evaluation=None)
        st.success(f"Loaded {len(bodies)} raw solids and generated {len(groups)} candidate component groups.")
    except Exception as e: st.exception(e)

if st.session_state.project is None:
    st.info("Upload a STEP file and click **Process STEP**.")
    st.stop()

p=st.session_state.project; groups=st.session_state.groups.copy(); bodies=st.session_state.bodies
m1,m2,m3,m4=st.columns(4); m1.metric("Raw solids",len(bodies)); m2.metric("Component groups",len(groups)); m3.metric("Assembly scale",f"{p['assembly_scale']:.1f}"); m4.metric("Motor axis",", ".join(f"{x:.2f}" for x in p["motor_axis"]))

tabs=st.tabs(["1 · STEP visualization","2 · Components","3 · Disassembly features","4 · Relationship matrix","5 · Sequence","6 · Interactive sequence","7 · Circularity evaluation","8 · Robotic assessment","9 · Improvement suggestions","10 · Export"])

with tabs[0]:
    st.subheader("Complete STEP assembly")
    st.plotly_chart(make_figure(p["meshes"],bodies,groups),use_container_width=True,config={"scrollZoom":True,"displaylogo":False})
    st.info("This stage is geometry-only. LLM reasoning starts at semantic interpretation stages; the original STEP geometry is never altered by the LLM.")

with tabs[1]:
    st.subheader("Component identification: geometry vs LLM vs user correction")
    c1,c2=st.columns(2)
    with c1:
        if st.button("Run LLM component identification",use_container_width=True):
            if not api_key: st.error("LLM is not configured on the server. Add OPENAI_API_KEY to Streamlit secrets or the server environment.")
            else:
                try:
                    st.session_state.groups=classify_with_openai(groups,api_key,model_name); groups=st.session_state.groups.copy(); st.success("LLM comparison completed.")
                except Exception as e: st.exception(e)
    with c2:
        gid=st.selectbox("Visualize one component",groups.group_id.astype(int).tolist(),format_func=lambda g:f"Group {g} · {groups.loc[groups.group_id==g,'final_component'].iloc[0]}")
    st.plotly_chart(make_figure(p["meshes"],bodies,groups,selected_group=int(gid),isolate=True),use_container_width=True,key="comp3d")
    edit=groups[["group_id","body_ids","heuristic_label","heuristic_confidence","api_label","api_confidence","final_component","user_label"]].copy()
    edit["body_ids"]=edit.body_ids.apply(lambda x:", ".join(map(str,x)))
    edited=st.data_editor(edit,use_container_width=True,hide_index=True,disabled=["group_id","body_ids","heuristic_label","heuristic_confidence","api_label","api_confidence","final_component"],column_config={"user_label":st.column_config.SelectboxColumn(options=["(keep automatic)"]+SEMANTIC_CLASSES)})
    if st.button("Apply component corrections",type="primary"):
        mapv=edited.set_index("group_id")["user_label"].to_dict()
        for idx,r in groups.iterrows():
            v=mapv.get(int(r.group_id)); groups.at[idx,"user_label"]=None if v in (None,"(keep automatic)","") else v
        st.session_state.groups=apply_user_labels(groups); groups=st.session_state.groups.copy(); st.success("Component corrections applied. Re-run downstream stages because semantic labels changed.")

with tabs[2]:
    st.subheader("Disassembly-feature extraction")
    st.markdown("**Each component:** geometry, holes/bores/slots candidates, dimensions, positions/orientations, and handling/inserting/fastening/mating function classes. Click a feature row to highlight its actual CAD face. Only the selected component is shown.")
    a,b=st.columns(2)
    with a:
        if st.button("Extract component assembly features",type="primary",use_container_width=True):
            try: st.session_state.features=extract_component_features(p,groups); st.success(f"Extracted {len(st.session_state.features)} component-level BRep features.")
            except Exception as e: st.exception(e)
    with b:
        if st.button("LLM reason about feature functions",use_container_width=True,disabled=st.session_state.features is None):
            if not api_key: st.error("LLM is not configured on the server. Add OPENAI_API_KEY to Streamlit secrets or the server environment.")
            else:
                try: st.session_state.features=reason_component_features(st.session_state.features,api_key,model_name); st.success("LLM feature-function comparison added.")
                except Exception as e: st.exception(e)
    feat=st.session_state.features
    if feat is not None and not feat.empty:
        group_name={int(r.group_id):str(r.final_component) for _,r in groups.iterrows()}
        gids=sorted(feat.group_id.unique().astype(int))
        gidf=st.selectbox(
            "Feature component",gids,key="feat_gid",
            format_func=lambda g:f"Group {g} · {group_name.get(int(g),'unknown')}"
        )
        local=feat[feat.group_id==gidf].copy().reset_index(drop=True)
        disp=local[["feature_id","body_id","feature_type","surface_type","function_class","dimension_1","dimension_2","dimension_3","position_x","position_y","position_z","orientation_x","orientation_y","orientation_z","confidence","llm_function_class","llm_reason","user_function_class","final_function_class"]].copy()
        st.caption("Click one row below. The corresponding BRep face and its feature type will be highlighted in the isolated 3D component view.")
        event=st.dataframe(
            disp,use_container_width=True,hide_index=True,key=f"feature_table_{gidf}",
            on_select="rerun",selection_mode="single-row"
        )
        selected_rows=list(event.selection.rows) if event is not None and hasattr(event,"selection") else []
        if selected_rows:
            selected=disp.iloc[int(selected_rows[0])]
            selected_id=int(selected.feature_id)
        else:
            selected=disp.iloc[0]
            selected_id=int(selected.feature_id)
            st.info("No row selected yet; showing the first feature. Click any table row to change the highlighted face.")

        selected_face_mesh=mesh_feature_face(
            p,int(selected.body_id),int(local.loc[local.feature_id==selected_id,"local_face_id"].iloc[0]),
            tolerance=max(mesh_tol*0.5,0.12)
        )
        feature_meshes={}
        if selected_face_mesh is not None:
            feature_meshes[(int(selected.body_id),int(local.loc[local.feature_id==selected_id,"local_face_id"].iloc[0]))]=selected_face_mesh
        st.plotly_chart(
            make_figure(
                p["meshes"],bodies,groups,selected_group=int(gidf),isolate=True,
                component_features=feat,selected_feature_ids=[selected_id],
                face_meshes=feature_meshes
            ),
            use_container_width=True,key=f"feat3d_{gidf}_{selected_id}",
            config={"scrollZoom":True,"displaylogo":False}
        )
        d1="—" if pd.isna(selected.dimension_1) else f"{float(selected.dimension_1):.3f}"
        d2="—" if pd.isna(selected.dimension_2) else f"{float(selected.dimension_2):.3f}"
        d3="—" if pd.isna(selected.dimension_3) else f"{float(selected.dimension_3):.3f}"
        st.write(f"**Selected feature F{selected_id}: `{selected.feature_type}`** · surface `{selected.surface_type}` · dimensions `{d1}, {d2}, {d3}` STEP units · final function `{selected.final_function_class}`")

        correction=st.text_input(
            "User correction for selected feature function",
            value="" if not isinstance(selected.user_function_class,str) else selected.user_function_class,
            key=f"feature_correction_{selected_id}",
            placeholder="Leave blank to keep geometry/LLM interpretation"
        )
        if st.button("Apply selected feature correction",key=f"apply_feature_{selected_id}"):
            out=feat.copy(); mask=out.feature_id==selected_id
            v=correction.strip()
            if v:
                out.loc[mask,"user_function_class"]=v; out.loc[mask,"final_function_class"]=v
            else:
                out.loc[mask,"user_function_class"]=None
                geom=out.loc[mask,"function_class"]
                llm=out.loc[mask,"llm_function_class"]
                out.loc[mask,"final_function_class"]=[lv if isinstance(lv,str) and lv.strip() else gv for gv,lv in zip(geom,llm)]
            st.session_state.features=out
            st.success(f"Feature F{selected_id} correction applied.")
        st.caption("The highlighted surface is the selected CAD face, not merely a centroid marker. Dimensions are raw STEP geometry units; normalized ML features are not shown as physical dimensions.")

with tabs[3]:
    st.subheader("Relationship matrix: component × component")
    st.markdown("Includes spatial adjacency, topology, interface, nominal clearance proxy, coaxiality, axial overlap, mating condition, fit interpretation, tolerance availability, precedence, and mapped component feature IDs.")
    a,b=st.columns(2)
    with a:
        if st.button("Extract pairwise relationships",type="primary",use_container_width=True,disabled=st.session_state.features is None):
            try: st.session_state.relationships=extract_relationships(p,groups,st.session_state.features); st.success(f"Generated {len(st.session_state.relationships)} pairwise relationships.")
            except Exception as e: st.exception(e)
    with b:
        if st.button("LLM reason about missing/ambiguous relationships",use_container_width=True,disabled=st.session_state.relationships is None):
            if not api_key: st.error("LLM is not configured on the server. Add OPENAI_API_KEY to Streamlit secrets or the server environment.")
            else:
                try: st.session_state.relationships=reason_relationships(st.session_state.relationships,api_key,model_name); st.success("LLM relationship comparison added.")
                except Exception as e: st.exception(e)
    rel=st.session_state.relationships
    if rel is not None:
        st.dataframe(relationship_matrix(groups,rel),use_container_width=True)
        st.warning("Fits/tolerances: geometric clearance can be screened, but true tolerance classes cannot be recovered unless PMI/GD&T exists in the STEP. The LLM may suggest a likely fit only; it must not fabricate numeric tolerances.")
        show=rel[["relationship_id","group_id_1","group_id_2","spatial_adjacency","topology_relation","interface_type","mating_condition","both_coaxial","axial_overlap_ratio","radial_clearance_estimate","fit_geometry","tolerance_status","precedence_hint","llm_interface_type","llm_fit","llm_precedence","user_interface_type","user_fit","user_precedence","final_interface_type","final_fit","final_precedence"]].copy()
        edr=st.data_editor(show,use_container_width=True,hide_index=True,disabled=[c for c in show.columns if c not in {"user_interface_type","user_fit","user_precedence"}])
        if st.button("Apply relationship corrections"):
            by=edr.set_index("relationship_id").to_dict("index"); out=rel.copy()
            for i,r in out.iterrows():
                x=by.get(int(r.relationship_id),{})
                for u,final,llm,geom in [("user_interface_type","final_interface_type","llm_interface_type","interface_type"),("user_fit","final_fit","llm_fit","fit_geometry"),("user_precedence","final_precedence","llm_precedence","precedence_hint")]:
                    v=x.get(u); lv=r.get(llm)
                    out.at[i,u]=v if isinstance(v,str) and v.strip() else None; out.at[i,final]=(v.strip() if isinstance(v,str) and v.strip() else (lv if isinstance(lv,str) and lv else r.get(geom)))
            st.session_state.relationships=out; st.success("Relationship corrections applied.")

with tabs[4]:
    st.subheader("Disassembly sequence: geometry vs LLM vs user")
    a,b=st.columns(2)
    with a:
        if st.button("Generate geometry-first sequence",type="primary",use_container_width=True,disabled=st.session_state.relationships is None):
            try:
                seq=generate_geometry_sequence(p,groups,st.session_state.relationships,st.session_state.features)
                # explicit schema guarantee fixes prior KeyError: ['group_id'] not in index
                for c in SEQUENCE_COLUMNS:
                    if c not in seq.columns: seq[c]=None
                st.session_state.sequence=seq[SEQUENCE_COLUMNS]; st.success(f"Generated {len(seq)} steps.")
            except Exception as e: st.exception(e)
    with b:
        if st.button("LLM review sequence",use_container_width=True,disabled=st.session_state.sequence is None):
            if not api_key: st.error("LLM is not configured on the server. Add OPENAI_API_KEY to Streamlit secrets or the server environment.")
            else:
                try: st.session_state.sequence=reason_sequence(st.session_state.sequence,st.session_state.relationships,api_key,model_name); st.success("LLM sequence comparison added.")
                except Exception as e: st.exception(e)
    seq=st.session_state.sequence
    if seq is not None:
        show=seq[["step","group_id","component","removal_direction","blocker_group_ids","tool","sequence_confidence","geometry_reason","llm_step","llm_direction","llm_confidence","llm_reason","user_step","user_direction","final_step","final_direction"]].copy()
        eds=st.data_editor(show,use_container_width=True,hide_index=True,disabled=[c for c in show.columns if c not in {"user_step","user_direction"}],column_config={"user_direction":st.column_config.SelectboxColumn(options=["","+motor_axis","-motor_axis"])})
        if st.button("Apply sequence corrections"):
            by=eds.set_index("group_id").to_dict("index"); out=seq.copy()
            for i,r in out.iterrows():
                x=by.get(int(r.group_id),{}); out.at[i,"user_step"]=x.get("user_step",np.nan); out.at[i,"user_direction"]=x.get("user_direction") or None
            st.session_state.sequence=apply_sequence_corrections(out); st.success("Sequence corrections applied.")

with tabs[5]:
    st.subheader("Interactive 3D disassembly sequence")
    seq=st.session_state.sequence
    if seq is None or seq.empty: st.info("Generate a sequence first.")
    else:
        ordered=seq.sort_values("final_step"); step=st.slider("Disassembly step",1,len(ordered),1)
        row=ordered.iloc[step-1]; gid=int(row.group_id); direction=np.asarray(p["motor_axis"],float)*(1 if row.final_direction=="+motor_axis" else -1)
        fids=[int(x) for x in row.assembly_feature_ids] if isinstance(row.assembly_feature_ids,(list,tuple)) else []
        explode={}
        for _,rr in ordered.iloc[:step-1].iterrows():
            d=np.asarray(p["motor_axis"],float)*(1 if rr.final_direction=="+motor_axis" else -1); explode[int(rr.group_id)]=(d,.35*float(p["assembly_scale"])*(1+.08*int(rr.final_step)))
        fig=make_figure(p["meshes"],bodies,groups,selected_group=gid,component_features=st.session_state.features,selected_feature_ids=fids,removal_direction=direction,assembly_scale=p["assembly_scale"],exploded_steps=explode)
        st.plotly_chart(fig,use_container_width=True,key="seq3d")
        st.write(f"**Step {int(row.final_step)}:** Group {gid} — `{row.component}` → `{row.final_direction}` using `{row.tool}`")
        st.write("**Geometry reason:**",row.geometry_reason)
        if isinstance(row.llm_reason,str) and row.llm_reason: st.write("**LLM comparison:**",row.llm_reason)

with tabs[6]:
    st.subheader("EV Motor Circularity Evaluator — human-confirmed assessment")
    st.info("This is the original supplied v5 evaluator, embedded without changing its scoring formulas. Enter and confirm design data, critical conditions and caps. Missing CAD/BOM information must not be guessed.")
    st.caption("The evaluator maintains its own browser-side state. Use its **Export assessment** button to save the completed JSON, then import that JSON below to link it to this project.")
    from streamlit.components.v1 import html as embed_html
    from pathlib import Path as _Path
    _evaluator_html = (_Path(__file__).resolve().parent / "circularity_evaluator.html").read_text(encoding="utf-8")
    embed_html(_evaluator_html, height=950, scrolling=True)
    assessment_upload = st.file_uploader("Import circularity evaluator JSON (exported using the embedded evaluator)", type=["json"], key="circularity_json_upload")
    if assessment_upload is not None and st.button("Attach circularity evaluation to project"):
        try:
            payload = json.loads(assessment_upload.getvalue().decode("utf-8"))
            if not isinstance(payload,dict) or not isinstance(payload.get("values"),dict) or not isinstance(payload.get("crit"),dict):
                raise ValueError("This is not a valid evaluator snapshot (expected values and crit objects).")
            st.session_state.circularity_evaluation = payload
            st.success("Circularity evaluator snapshot attached to this project and included in export.")
        except Exception as exc:
            st.error(f"Cannot import assessment: {exc}")
    if st.session_state.get("circularity_evaluation"):
        snapshot=st.session_state.circularity_evaluation
        st.success(f"Saved evaluator inputs: {len(snapshot.get('values',{}))}; critical-condition categories: {len(snapshot.get('crit',{}))}.")
        st.download_button("Download attached circularity snapshot", json.dumps(snapshot,indent=2), "circularity_evaluation.json", "application/json")

with tabs[7]:
    st.subheader("Robotic disassembly assessment")
    st.write(f"Selected existing system: **{ROBOT_SYSTEM['name']}**")
    c1,c2,c3,c4=st.columns(4); c1.metric("Payload",f"{ROBOT_SYSTEM['payload_kg']} kg"); c2.metric("Reach",f"{ROBOT_SYSTEM['reach_mm']} mm"); c3.metric("Repeatability",f"±{ROBOT_SYSTEM['repeatability_mm']} mm"); c4.metric("Typical robot power",f"{ROBOT_SYSTEM['typical_power_w']} W")
    a,b,c=st.columns(3); toolchange=a.number_input("Assumed automatic tool-change time (s)",1.0,30.0,5.0,.5); speed=b.number_input("Assessment TCP speed (m/s)",.03,1.0,.20,.01); fixture=c.toggle("Assembly clamped in fixture",value=True)
    if st.button("Run robotic assessment",type="primary",disabled=st.session_state.sequence is None):
        st.session_state.assessment=assess_disassembly(p,groups,apply_sequence_corrections(st.session_state.sequence),st.session_state.relationships,toolchange,speed,fixture); st.success("Robotic screening assessment complete.")
    ass=st.session_state.assessment
    if ass is not None:
        st.dataframe(ass,use_container_width=True,hide_index=True)
        q1,q2,q3,q4=st.columns(4); q1.metric("Total disassembly time",f"{ass.estimated_disassembly_time_s.sum()+ass.tool_change_time_s.sum():.1f} s"); q2.metric("Tool changes",int((ass.tool_change_time_s>0).sum())); q3.metric("Estimated robot energy",f"{ass.estimated_energy_Wh.sum():.2f} Wh"); q4.metric("Max disassembly depth",int(ass.disassembly_depth.max()))
        st.caption("These are parametric engineering screening estimates, not robot simulation. Stability uses a fixture/sequence heuristic; material recovery is limited because material metadata is not available in the geometric STEP evidence.")
        if st.button("LLM review robotic assessment", use_container_width=True):
            if not api_key: st.error("LLM is not configured on the server. Add OPENAI_API_KEY to Streamlit secrets or the server environment.")
            else:
                try:
                    st.session_state.assessment=reason_assessment(ass,api_key,model_name); ass=st.session_state.assessment; st.success("LLM assessment comparison added.")
                except Exception as e: st.exception(e)
        if "llm_reason" in ass.columns:
            cols=[c for c in ["final_step","group_id","component","robot_feasibility","stability_risk","tooling_difficulty","llm_feasibility","llm_risk","llm_reason","user_assessment","final_assessment"] if c in ass.columns]
            ea=st.data_editor(ass[cols],use_container_width=True,hide_index=True,disabled=[c for c in cols if c!="user_assessment"])
            if st.button("Apply robotic-assessment corrections"):
                by=ea.set_index("group_id")["user_assessment"].to_dict(); out=ass.copy()
                for i,r in out.iterrows():
                    v=by.get(int(r.group_id));
                    if isinstance(v,str) and v.strip(): out.at[i,"user_assessment"]=v.strip(); out.at[i,"final_assessment"]=v.strip()
                st.session_state.assessment=out; st.success("Assessment corrections applied.")

with tabs[8]:
    st.subheader("Design-specific suggestions for reuse, remanufacturing, recycling, recovery, repairability and refurbishment")
    st.info("Recommendations are generated only when an extracted feature, relationship, sequence step, robotic-assessment result, or file-specific data gap supports them. No evidence → no recommendation.")
    if st.button("Generate design-specific recommendations",type="primary",disabled=st.session_state.assessment is None):
        st.session_state.suggestions=improvement_suggestions(st.session_state.assessment,st.session_state.relationships,st.session_state.features)
        st.success(f"Generated {len(st.session_state.suggestions)} evidence-linked recommendations.")
    sug=st.session_state.suggestions
    if sug is not None:
        if sug.empty:
            st.warning("No recommendation met the evidence rules for this design.")
        else:
            view_cols=["recommendation_id","priority","category","group_id","component","final_step","relationship_ids","feature_ids","current_issue","measured_evidence","design_change","expected_disassembly_effect","circularity_benefit","confidence","llm_suggestion","llm_reason","user_correction","final_recommendation"]
            st.dataframe(sug[[c for c in view_cols if c in sug.columns]],use_container_width=True,hide_index=True)
            if st.button("LLM review design-specific recommendations",disabled=not api_key):
                try:
                    st.session_state.suggestions=reason_improvements(sug,st.session_state.assessment,st.session_state.relationships,st.session_state.features,api_key,model_name)
                    sug=st.session_state.suggestions; st.success("LLM comparison completed. LLM output remains separate from the deterministic CAD recommendation until accepted/corrected by the user.")
                except Exception as e: st.exception(e)
            elif not api_key:
                st.caption("LLM review is disabled because OPENAI_API_KEY is not configured on the server.")

            ids=sug.recommendation_id.astype(int).tolist()
            rid=st.selectbox("Review/correct recommendation",ids,format_func=lambda x:f"R{x} · {sug.loc[sug.recommendation_id==x,'component'].iloc[0]} · {sug.loc[sug.recommendation_id==x,'category'].iloc[0]}")
            row=sug[sug.recommendation_id==rid].iloc[0]
            st.write("**CAD/assessment recommendation:**",row.design_change)
            if isinstance(row.llm_suggestion,str) and row.llm_suggestion:
                st.write("**LLM comparison:**",row.llm_suggestion)
                st.caption(str(row.llm_reason) if isinstance(row.llm_reason,str) else "")
            user_rec=st.text_area("User correction / accepted final recommendation",value="" if not isinstance(row.user_correction,str) else row.user_correction,key=f"rec_corr_{rid}")
            c1,c2=st.columns(2)
            if c1.button("Use user correction",key=f"rec_user_{rid}"):
                out=sug.copy(); mask=out.recommendation_id==rid; v=user_rec.strip()
                if v:
                    out.loc[mask,"user_correction"]=v; out.loc[mask,"final_recommendation"]=v
                    st.session_state.suggestions=out; st.success("User recommendation stored as final.")
                else: st.warning("Enter a correction first.")
            if c2.button("Accept LLM comparison",key=f"rec_llm_{rid}",disabled=not (isinstance(row.llm_suggestion,str) and row.llm_suggestion)):
                out=sug.copy(); mask=out.recommendation_id==rid; out.loc[mask,"final_recommendation"]=row.llm_suggestion; st.session_state.suggestions=out; st.success("LLM recommendation accepted as final for this evidence record.")
            st.caption("Every final recommendation remains traceable to its component/group, relationship IDs, feature IDs, measured evidence and sequence/assessment trigger.")

with tabs[9]:
    st.subheader("Export audit trail")
    datasets={"components":st.session_state.groups,"component_features":st.session_state.features,"relationships":st.session_state.relationships,"disassembly_sequence":st.session_state.sequence,"robotic_assessment":st.session_state.assessment,"improvement_suggestions":st.session_state.suggestions}
    for name,df in datasets.items():
        if df is not None:
            st.download_button(f"Download {name}.csv",df.to_csv(index=False).encode(),file_name=f"{name}.csv",mime="text/csv",key=f"dl_{name}")
    audit={k:(None if v is None else v.to_dict("records")) for k,v in datasets.items()}
    audit["circularity_evaluation"] = st.session_state.get("circularity_evaluation")
    st.download_button("Download complete audit JSON",json.dumps(audit,default=str,indent=2).encode(),file_name="ev_motor_disassembly_audit.json",mime="application/json")
