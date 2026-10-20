from __future__ import annotations
import numpy as np
import plotly.graph_objects as go

COLORS={"rotor":"#e76f51","rotor_core":"#f4a261","stator":"#2a9d8f","shaft":"#6c757d","bearing":"#457b9d","motor_housing":"#adb5bd","front_end_plate":"#9b5de5","rear_end_plate":"#8338ec","end_plate_or_end_cap":"#9b5de5","encoder":"#06d6a0","sensor":"#00b4d8","magnet":"#ef233c","fastener":"#8d99ae","cooling_component":"#48cae4","electrical_connector":"#ffbe0b","repeated_internal_element":"#fb8500","other":"#bc6c25","unknown":"#d3d3d3"}

FEATURE_COLORS={
    "hole_or_bore_candidate":"#ff006e",
    "slot_candidate":"#8338ec",
    "cylindrical_mating_surface":"#fb5607",
    "conical_mating_surface":"#ffbe0b",
    "planar_face":"#3a86ff",
    "spherical_surface":"#00b4d8",
    "toroidal_surface":"#06d6a0",
    "freeform_surface":"#ef233c",
    "surface":"#ef233c",
}


def _add_body(fig,mesh,color,name,opacity=1.0,translation=None):
    p=np.asarray(mesh["points"],float).copy(); tri=np.asarray(mesh["triangles"],int)
    if translation is not None: p += np.asarray(translation,float)
    if len(p)==0 or len(tri)==0:return
    fig.add_trace(go.Mesh3d(x=p[:,0],y=p[:,1],z=p[:,2],i=tri[:,0],j=tri[:,1],k=tri[:,2],color=color,opacity=opacity,name=name,hovertemplate=name+"<extra></extra>",flatshading=False,showscale=False))


def _feature_hover(row):
    vals=[]
    for col,label in [("dimension_1","D1"),("dimension_2","D2"),("dimension_3","D3")]:
        try:
            v=float(row.get(col,np.nan))
            if np.isfinite(v): vals.append(f"{label}={v:.3f}")
        except Exception: pass
    dims=" · ".join(vals) if vals else "dimensions unavailable"
    return f"F{int(row.feature_id)} · {row.feature_type}<br>{row.surface_type}<br>{dims}"


def make_figure(meshes,bodies,groups,selected_group=None,isolate=False,component_features=None,selected_feature_ids=None,removal_direction=None,assembly_scale=None,exploded_steps=None,face_meshes=None):
    fig=go.Figure(); label={int(r.group_id):str(r.final_component) for _,r in groups.iterrows()}; body_group={int(r.body_id):int(r.group_id) for _,r in bodies.iterrows()}
    selected_feature_ids=set(int(x) for x in (selected_feature_ids or [])); exploded_steps=exploded_steps or {}; face_meshes=face_meshes or {}
    for bid,mesh in meshes.items():
        gid=body_group.get(int(bid)); lab=label.get(gid,"unknown")
        if isolate and selected_group is not None and gid!=selected_group: continue
        op=1.0 if selected_group is None or gid==selected_group else .12
        trans=None
        if gid in exploded_steps:
            direction,mag=exploded_steps[gid]; trans=np.asarray(direction,float)*float(mag)
        _add_body(fig,mesh,COLORS.get(lab,"#d3d3d3"),f"Body {bid} · G{gid} · {lab}",op,trans)

    # Selected feature: overlay its actual tessellated BRep face and retain a centroid marker.
    if component_features is not None and len(component_features) and selected_feature_ids:
        q=component_features[component_features.feature_id.isin(selected_feature_ids)].copy()
        for _,r in q.iterrows():
            ftype=str(r.feature_type); fcolor=FEATURE_COLORS.get(ftype,"#ff1744")
            key=(int(r.body_id),int(r.local_face_id))
            fmesh=face_meshes.get(key)
            hover=_feature_hover(r)
            if fmesh is not None:
                pts=np.asarray(fmesh["points"],float); tri=np.asarray(fmesh["triangles"],int)
                if len(pts) and len(tri):
                    fig.add_trace(go.Mesh3d(
                        x=pts[:,0],y=pts[:,1],z=pts[:,2],i=tri[:,0],j=tri[:,1],k=tri[:,2],
                        color=fcolor,opacity=.96,name=f"Feature: {ftype}",
                        hovertemplate=hover+"<extra></extra>",flatshading=False,showscale=False,
                        lighting={"ambient":0.8,"diffuse":0.8,"specular":0.35},
                    ))
            fig.add_trace(go.Scatter3d(
                x=[float(r.position_x)],y=[float(r.position_y)],z=[float(r.position_z)],mode="markers+text",
                marker={"size":6,"color":fcolor,"symbol":"diamond","line":{"width":2,"color":"black"}},
                text=[ftype],textposition="top center",name=f"F{int(r.feature_id)} · {ftype}",
                hovertemplate=hover+"<extra></extra>"
            ))

    if selected_group is not None and removal_direction is not None and assembly_scale:
        ids=bodies[bodies.group_id==selected_group]; c=ids[["centroid_x","centroid_y","centroid_z"]].to_numpy(float).mean(axis=0); d=np.asarray(removal_direction,float); d=d/max(np.linalg.norm(d),1e-9); end=c+d*.25*float(assembly_scale)
        fig.add_trace(go.Scatter3d(x=[c[0],end[0]],y=[c[1],end[1]],z=[c[2],end[2]],mode="lines+markers",line={"width":8,"color":"red"},name="Removal direction"))
    fig.update_layout(scene={"aspectmode":"data","xaxis_title":"X","yaxis_title":"Y","zaxis_title":"Z"},margin={"l":0,"r":0,"t":30,"b":0},height=680,legend={"itemsizing":"constant"})
    return fig
