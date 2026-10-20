# EV Motor Disassembly Intelligence Platform

A Streamlit web application for geometry-grounded EV-motor disassembly analysis.

## Workflow
1. Upload STEP/STP
2. Visualize full assembly
3. Identify components (geometry / optional LLM / user correction)
4. Extract per-component disassembly features
5. Build component-to-component relationship matrix
6. Generate disassembly sequence
7. Visualize sequence interactively in 3D
8. Assess sequence for a robotic cell
9. Generate circularity/serviceability improvements
10. Export full audit trail

## Important modelling boundaries
- Original assembly names/hierarchy are not assumed.
- Hole/slot/fit semantics are candidates inferred from BRep geometry; geometry evidence and semantic interpretation are stored separately.
- Actual ISO fits/tolerances require PMI/GD&T or other manufacturing metadata. The app does not invent numeric tolerances when they are absent.
- Robotic times, stability and energy are screening estimates, not a dynamic robot simulation.
- Material recovery cannot be calculated reliably unless material metadata is supplied.

## Robotic reference cell
Default assessment profile: Universal Robots UR10e + ATI QC-11 automatic tool changer + Robotiq adaptive 2-finger gripper.
The app uses a 12.5 kg UR10e payload, 1300 mm reach, ±0.05 mm pose repeatability and 350 W typical-program power as reference values. Tool-change time and TCP speed are editable assumptions.

## Install
Recommended Python: 3.11 or 3.12.

Windows Command Prompt:
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open `http://localhost:8501`.

## Optional OpenAI reasoning
The OpenAI key is deliberately hidden from the browser UI. Configure it only on the machine/server running Streamlit. Either set the `OPENAI_API_KEY` environment variable, or copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and place the key there. Never commit `secrets.toml`. The sidebar shows only whether LLM access is configured. The application stores geometry output, LLM output, user correction, and final value side-by-side.


## Revision: feature selection and design-specific recommendations

Section 3 uses Streamlit row selection. Select a component from a dropdown shown as `Group ID · component name`, then click one feature row. Only that component is rendered and the selected BRep face is overlaid in the 3D view with its feature type and dimensions.

The recommendation stage is evidence-gated. Every deterministic recommendation must reference one or more concrete triggers from this uploaded design: a component/group, disassembly step, relationship ID, feature ID, tooling-difficulty result, stability result, tool-change event, or a file-specific missing-data condition. If no extracted evidence supports a recommendation, no recommendation is generated. LLM review is constrained to the same evidence and is required to return `Insufficient design evidence` rather than invent unsupported geometry, materials, fits, tolerances, or fasteners.


## Circularity evaluator (new Section 7)
The original uploaded `EV_Motor_Circularity_Evaluator_Prototype_v5_Usability.html` is bundled as `circularity_evaluator.html` and embedded in the Streamlit app before robotic assessment. It retains the original JavaScript calculations, 11 category inputs, critical conditions, caps, evidence, and redesign guidance. The embedded page is a separate browser document: use **Export assessment** inside it, then upload the JSON in Section 7 and click **Attach circularity evaluation to project**. This snapshot is included in the complete audit JSON. The HTML's local browser storage is not guaranteed to persist across Streamlit rerenders; use the JSON export for reliable persistence. CAD fields are not automatically assigned to subjective questionnaire inputs; confirm values manually and mark unavailable evidence rather than inventing it.
