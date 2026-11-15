import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="UACR Research Prediction Prototype", page_icon="🧪", layout="centered")

st.title("UACR Research Prediction Prototype")
st.caption("NHANES 2017–2018 liver–muscle cohort | Paired Model A vs Model B")

st.error(
    "RESEARCH PROTOTYPE ONLY — NOT FOR CLINICAL DECISION-MAKING. "
    "This model was developed from a selected US NHANES-derived cohort and has not been "
    "externally validated in your clinic or South Indian patients. Do not use its output "
    "to diagnose, rule out, defer or determine treatment. Laboratory UACR and clinician "
    "assessment remain necessary."
)

MODEL_PATH = "uacr_modelAB_bundle.joblib"
if not os.path.exists(MODEL_PATH):
    st.warning(
        "Model bundle not found. Run the Google Colab notebook first, download "
        "NHANES_UACR_ModelAB_Results.zip, extract uacr_modelAB_bundle.joblib, "
        "and place it beside this app.py file."
    )
    st.stop()

bundle = joblib.load(MODEL_PATH)

st.subheader("Enter participant variables")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age (years)", min_value=20, max_value=59, value=40, step=1)
    sex_label = st.selectbox("Sex", ["Male", "Female"])
    bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=80.0, value=27.0, step=0.1)
    waist = st.number_input("Waist circumference (cm)", min_value=40.0, max_value=180.0, value=90.0, step=0.1)
with c2:
    hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=20.0, value=5.7, step=0.1)
    muscle_mass = st.number_input("Muscle mass (kg)", min_value=3.0, max_value=80.0, value=22.0, step=0.1)

sex_code = 1 if sex_label == "Male" else 2
muscle_index = round(muscle_mass / bmi, 3) if bmi > 0 else np.nan

st.caption(
    f"Muscle-to-BMI index calculated as muscle mass ÷ BMI = {muscle_index:.3f}. "
    "This matches the uploaded worksheet's index construction to three decimal places; "
    "confirm source documentation before clinical use."
)

if st.button("Generate research predictions", type="primary", use_container_width=True):
    row_A = pd.DataFrame([{
        "Age_Years": age,
        "Sex_1M_2F": sex_code,
        "BMI": bmi,
        "Waist_cm": waist,
        "HbA1c_pct": hba1c
    }])
    row_B = row_A.copy()
    row_B["Muscle_to_BMI_Index"] = muscle_index

    pA = float(bundle["model_A"].predict_proba(row_A)[0, 1])
    pB = float(bundle["model_B"].predict_proba(row_B)[0, 1])
    tA = float(bundle["model_A_threshold_youden"])
    tB = float(bundle["model_B_threshold_youden"])

    st.subheader("Model outputs")
    left, right = st.columns(2)
    with left:
        st.metric("Model A — no muscle", f"{pA*100:.1f}%")
        st.caption(f"Selected algorithm: {bundle['model_A_algorithm']}")
        st.caption(f"Development threshold: {tA:.3f}")
        st.write("Above development threshold" if pA >= tA else "Below development threshold")
    with right:
        st.metric("Model B — muscle added", f"{pB*100:.1f}%")
        st.caption(f"Selected algorithm: {bundle['model_B_algorithm']}")
        st.caption(f"Development threshold: {tB:.3f}")
        st.write("Above development threshold" if pB >= tB else "Below development threshold")

    st.info(
        "These are model-estimated probabilities, not measured UACR values or a diagnosis. "
        "The Youden thresholds were selected in development data and are not clinically validated. "
        "A higher Model B probability does not prove muscle dysfunction caused albuminuria."
    )

    st.subheader("Entered values")
    st.dataframe(pd.DataFrame({
        "Variable": ["Age", "Sex", "BMI", "Waist", "HbA1c", "Muscle mass", "Derived muscle-to-BMI index"],
        "Value": [age, sex_label, bmi, waist, hba1c, muscle_mass, muscle_index]
    }), use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Research use only. Not a medical device. No patient data are stored by this app code, "
    "but do not enter identifiable patient information into a public deployment."
)
