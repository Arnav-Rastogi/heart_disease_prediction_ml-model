from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.prediction import ModelFilesError, load_artifacts, predict_patient
from utils.report import build_pdf_report


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "heart_disease_model.pkl"
FEATURE_PATH = ROOT / "feature_columns.pkl"
SCALER_PATH = ROOT / "scaler.pkl"
DATA_PATH = ROOT / "heart.csv"

st.set_page_config(page_title="CardioPredict AI", page_icon="♥", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --navy:#10243a; --navy2:#17334f; --red:#e84d5b; --muted:#718096; --line:#e7ecf2; --paper:#f5f7fa; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--paper); color:var(--navy); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#10243a 0%,#17334f 100%); }
[data-testid="stSidebar"] * { color:#edf4fb !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius:10px; padding:8px 10px; }
.hero { background:linear-gradient(120deg,#10243a,#1a3a58 65%,#244966); border-radius:22px; padding:34px 38px; color:white; position:relative; overflow:hidden; box-shadow:0 16px 36px rgba(16,36,58,.14); }
.hero:after { content:'♥'; position:absolute; right:40px; top:-40px; font-size:230px; color:rgba(255,255,255,.05); }
.eyebrow { color:#ff9ca4; text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:700; }
.hero h1 { font-family:Manrope,sans-serif; font-size:37px; line-height:1.15; margin:10px 0; color:#fff; }
.hero p { color:#d2dfec; max-width:650px; font-size:15px; }
.section-title { font-family:Manrope,sans-serif; font-weight:800; font-size:22px; color:#10243a; margin:26px 0 14px; }
.metric-card { background:white; border:1px solid var(--line); border-radius:16px; padding:19px 21px; min-height:115px; box-shadow:0 6px 18px rgba(19,42,66,.045); }
.metric-label { color:var(--muted); font-size:12px; font-weight:600; }
.metric-value { color:var(--navy); font-family:Manrope,sans-serif; font-size:28px; font-weight:800; margin-top:5px; }
.metric-foot { color:#91a0b1; font-size:11px; margin-top:3px; }
.info-card { background:#fff; border:1px solid var(--line); padding:20px 22px; border-radius:16px; height:100%; box-shadow:0 6px 18px rgba(19,42,66,.04); }
.info-card h3 { font:700 16px Manrope,sans-serif; color:var(--navy); margin:0 0 8px; }
.info-card p { color:#6f7f91; font-size:13px; line-height:1.6; margin:0; }
.stButton > button, .stDownloadButton > button { border-radius:10px; min-height:43px; font-weight:700; }
.stButton > button[kind="primary"] { background:var(--red); border-color:var(--red); }
.disclaimer { color:#738295; background:#fff; border:1px solid var(--line); padding:13px 16px; border-radius:12px; font-size:12px; }
div[data-testid="stForm"] { background:#fff; padding:24px; border:1px solid var(--line); border-radius:18px; }
@media(max-width:700px) { .hero{padding:26px 22px}.hero h1{font-size:29px}.hero:after{right:-20px;font-size:170px} }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the original trained model…")
def get_artifacts(model_path: str, feature_path: str, scaler_path: str):
    return load_artifacts(model_path, feature_path, scaler_path)


@st.cache_data
def load_dataset(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None


def metric_card(label: str, value: str, foot: str = ""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>', unsafe_allow_html=True)


def show_hero(eyebrow: str, title: str, description: str):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>', unsafe_allow_html=True)


try:
    model, feature_columns, scaler = get_artifacts(str(MODEL_PATH), str(FEATURE_PATH), str(SCALER_PATH))
except ModelFilesError as exc:
    st.error(f"The saved prediction files could not be loaded. {exc}")
    st.info("Place heart_disease_model.pkl, feature_columns.pkl, and scaler.pkl beside app.py, then refresh the page.")
    st.stop()

data = load_dataset(str(DATA_PATH))

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

with st.sidebar:
    st.markdown("<div style='font:800 21px Manrope;color:white;padding:16px 2px 4px'>♥ CardioPredict <span style='color:#ff9ca4'>AI</span></div><div style='color:#a9bfd2;font-size:11px;letter-spacing:1.3px;text-transform:uppercase;margin-bottom:25px'>Heart health intelligence</div>", unsafe_allow_html=True)
    page = st.radio("Navigation", ["Dashboard", "Heart Health Prediction", "Prediction Results", "Model Analytics"],
                    index=["Dashboard", "Heart Health Prediction", "Prediction Results", "Model Analytics"].index(st.session_state.page), label_visibility="collapsed")
    st.session_state.page = page
    st.markdown("<div style='height:24px'></div><div style='border-top:1px solid #ffffff20;padding-top:16px;color:#a9bfd2;font-size:11px;line-height:1.7'>MODEL<br><b style='color:white'>Logistic Regression</b><br><br>TRAINING NOTE<br><b style='color:white'>918 patient records</b></div>", unsafe_allow_html=True)

if page == "Dashboard":
    show_hero("CARDIOVASCULAR INSIGHT · MADE CLEAR", "A clearer view of heart health.",
              "Explore a machine learning risk estimate built from patient health indicators. Thoughtful inputs, transparent model context, and a simple, polished experience.")
    st.markdown('<div class="section-title">Model at a glance</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, values in zip(cols, [("Test accuracy", "86.8%", "Notebook holdout result"), ("Algorithm", "Logistic", "Regression classifier"),
                                  ("Training records", f"{len(data):,}" if data is not None else "918", "Heart dataset observations"),
                                  ("Model inputs", str(len(feature_columns)), "Encoded feature columns")]):
        with col: metric_card(*values)
    st.markdown('<div class="section-title">Designed for a more informed conversation</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    for col, title, body in [(a, "01 · Share the indicators", "Enter 11 patient measurements and categories in a guided, validated form."),
                             (b, "02 · Review the estimate", "See the model output alongside class probabilities and a summary of the submitted inputs."),
                             (c, "03 · Keep the context", "Download a concise PDF report with the output and a clear informational-use notice.")]:
        with col: st.markdown(f'<div class="info-card"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Start a prediction →", type="primary", use_container_width=False):
        st.session_state.page = "Heart Health Prediction"
        st.rerun()
    st.markdown('<div class="disclaimer">For educational and informational purposes only. This model output is not a medical diagnosis or a substitute for care from a qualified professional.</div>', unsafe_allow_html=True)

elif page == "Heart Health Prediction":
    show_hero("PATIENT PROFILE", "Enter health indicators", "Complete each field to generate an estimate from the original trained model. All values are processed using the saved training scaler.")
    st.markdown('<div class="section-title">Patient information</div>', unsafe_allow_html=True)
    with st.form("patient_form"):
        st.caption("All 11 original model inputs are required.")
        row1 = st.columns(4)
        with row1[0]: age = st.number_input("Age · years", min_value=18, max_value=100, value=54, step=1)
        with row1[1]: sex = st.selectbox("Sex", ["F", "M"], format_func=lambda x: "Female" if x == "F" else "Male")
        with row1[2]: chest = st.selectbox("Chest pain type", ["ASY", "ATA", "NAP", "TA"], format_func=lambda x: {"ASY":"Asymptomatic", "ATA":"Atypical angina", "NAP":"Non-anginal pain", "TA":"Typical angina"}[x])
        with row1[3]: bp = st.number_input("Resting blood pressure · mmHg", min_value=0, max_value=300, value=130, step=1, help="Zero is permitted because it occurs in the source dataset.")
        row2 = st.columns(4)
        with row2[0]: chol = st.number_input("Cholesterol · mg/dL", min_value=0, max_value=1000, value=220, step=1, help="Zero is permitted because it occurs in the source dataset.")
        with row2[1]: fasting = st.selectbox("Fasting blood sugar > 120 mg/dL", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        with row2[2]: ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"], format_func=lambda x: {"Normal":"Normal", "ST":"ST-T wave abnormality", "LVH":"Left ventricular hypertrophy"}[x])
        with row2[3]: maxhr = st.number_input("Maximum heart rate · bpm", min_value=0, max_value=300, value=145, step=1)
        row3 = st.columns(3)
        with row3[0]: angina = st.selectbox("Exercise-induced angina", ["N", "Y"], format_func=lambda x: "No" if x == "N" else "Yes")
        with row3[1]: oldpeak = st.number_input("Oldpeak · ST depression", min_value=-10.0, max_value=20.0, value=0.0, step=0.1, format="%.1f")
        with row3[2]: slope = st.selectbox("ST slope", ["Up", "Flat", "Down"], format_func=lambda x: {"Up":"Upsloping", "Flat":"Flat", "Down":"Downsloping"}[x])
        submitted = st.form_submit_button("Generate estimate", type="primary", use_container_width=True)
    if submitted:
        patient = {"Age": age, "Sex": sex, "ChestPainType": chest, "RestingBP": bp, "Cholesterol": chol,
                   "FastingBS": fasting, "RestingECG": ecg, "MaxHR": maxhr, "ExerciseAngina": angina,
                   "Oldpeak": oldpeak, "ST_Slope": slope}
        try:
            result = predict_patient(patient, model, feature_columns, scaler)
            st.session_state.last_prediction = {"patient": patient, "prediction": result["prediction"],
                                                "probability_disease": result["probability_disease"],
                                                "probability_no_disease": result["probability_no_disease"]}
            st.session_state.page = "Prediction Results"
            st.rerun()
        except (ValueError, ModelFilesError) as exc:
            st.error(f"Unable to process these values: {exc}")
    st.markdown('<div class="disclaimer">This estimate is informational only and should not be used for diagnosis or treatment decisions.</div>', unsafe_allow_html=True)

elif page == "Prediction Results":
    show_hero("YOUR MODEL OUTPUT", "Prediction results", "A probability estimate from the saved Logistic Regression model, with the patient inputs kept in view.")
    result = st.session_state.last_prediction
    if result is None:
        st.info("There is no prediction to display yet. Enter patient information to generate an estimate.")
        if st.button("Go to patient form", type="primary"):
            st.session_state.page = "Heart Health Prediction"
            st.rerun()
    else:
        disease = result["probability_disease"]
        st.markdown('<div class="section-title">Estimated probabilities</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.5])
        with c1:
            metric_card("HEART DISEASE CLASS PROBABILITY", f"{disease:.1%}", "Model probability · not a diagnosis")
            st.markdown("<br>", unsafe_allow_html=True)
            if result["prediction"] == 1:
                st.warning("The model assigned this profile to its heart disease class. Please review the estimate with a qualified healthcare professional.")
            else:
                st.success("The model assigned this profile to its no heart disease class. This result does not rule out a health condition.")
        with c2:
            fig = go.Figure(go.Bar(x=[result["probability_no_disease"], disease], y=["No heart disease class", "Heart disease class"],
                                   orientation="h", marker_color=["#92a6bb", "#e84d5b"], text=[f"{result['probability_no_disease']:.1%}", f"{disease:.1%}"], textposition="auto"))
            fig.update_layout(height=230, margin=dict(l=10,r=20,t=15,b=10), xaxis=dict(range=[0,1], tickformat=".0%", title=None, gridcolor="#edf1f5"),
                              yaxis_title=None, paper_bgcolor="white", plot_bgcolor="white", font=dict(family="DM Sans", color="#10243a"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('<div class="section-title">Patient input summary</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([result["patient"]]).T.rename(columns={0:"Submitted value"}), use_container_width=True)
        st.download_button("Download PDF report", data=build_pdf_report(result["patient"], result), file_name="cardiopredict_report.pdf", mime="application/pdf", type="primary")
        st.markdown('<br><div class="disclaimer">This model output is not a medical diagnosis, does not replace professional medical advice, and should not be used to make healthcare decisions.</div>', unsafe_allow_html=True)

else:
    show_hero("MODEL TRANSPARENCY", "Performance & dataset", "Review the results reported by the original notebook and understand the dataset used to develop this model.")
    st.markdown('<div class="section-title">Reported holdout performance</div>', unsafe_allow_html=True)
    # These metrics are derived directly from the notebook's saved test confusion matrix.
    tn, fp, fn, tp = 111, 12, 28, 152
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = 2 * precision * recall / (precision + recall)
    vals = [("Accuracy", f"{accuracy:.1%}", "Notebook: 86.8%"), ("Precision", f"{precision:.1%}", "Derived from confusion matrix"),
            ("Recall · sensitivity", f"{recall:.1%}", "Derived from confusion matrix"), ("F1 score", f"{f1:.1%}", "Derived from confusion matrix"),
            ("Specificity", f"{specificity:.1%}", "Derived from confusion matrix")]
    cols = st.columns(5)
    for col, values in zip(cols, vals):
        with col: metric_card(*values)
    st.caption("The notebook reports 86.7987% accuracy and confusion matrix [[111, 12], [28, 152]] in negative/positive label order. Other metrics above are calculated from those same counts.")
    st.markdown('<div class="section-title">Confusion matrix</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1.25])
    with c1:
        cm = go.Figure(go.Heatmap(z=[[tn, fp],[fn,tp]], x=["Predicted no disease", "Predicted disease"], y=["Actual no disease", "Actual disease"],
                                  colorscale=[[0,"#e8eff5"],[0.5,"#7590a8"],[1,"#10243a"]], text=[[tn,fp],[fn,tp]], texttemplate="%{text}", showscale=False))
        cm.update_layout(height=330, margin=dict(l=5,r=5,t=15,b=5), paper_bgcolor="white", plot_bgcolor="white", font=dict(family="DM Sans", color="#10243a"))
        st.plotly_chart(cm, use_container_width=True, config={"displayModeBar":False})
    with c2:
        st.markdown('<div class="section-title" style="margin-top:0">Dataset overview</div>', unsafe_allow_html=True)
        if data is not None:
            d1,d2,d3 = st.columns(3)
            with d1: metric_card("RECORDS", f"{len(data):,}", "Source heart.csv")
            with d2: metric_card("INPUT VARIABLES", "11", "Original patient inputs")
            with d3: metric_card("TARGET", "Binary", "HeartDisease · 0 / 1")
            st.markdown("<br>", unsafe_allow_html=True)
            counts = data["HeartDisease"].value_counts().sort_index()
            fig = px.pie(values=counts.values, names=["No disease class" if x == 0 else "Disease class" for x in counts.index], hole=.65,
                         color_discrete_sequence=["#92a6bb", "#e84d5b"])
            fig.update_layout(height=260, margin=dict(l=5,r=5,t=5,b=5), paper_bgcolor="white", legend=dict(orientation="h",y=-.1), font=dict(family="DM Sans",color="#10243a"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        else:
            st.warning("heart.csv is unavailable. Dataset visualizations cannot be displayed.")
    st.markdown('<div class="section-title">Feature groups</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({"Input field": ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"],
                               "Type": ["Numeric", "Categorical", "Categorical", "Numeric", "Numeric", "Binary", "Categorical", "Numeric", "Binary", "Numeric", "Categorical"],
                               "Model handling": ["Standardized"]*1+["One-hot encoded", "One-hot encoded", "Standardized", "Standardized", "Binary indicator", "One-hot encoded", "Standardized", "One-hot encoded", "Standardized", "One-hot encoded"]}),
                 use_container_width=True, hide_index=True)
    st.caption("Preprocessing matches the notebook: pandas get_dummies(drop_first=True), then the saved StandardScaler on Age, RestingBP, Cholesterol, MaxHR, and Oldpeak. The test set was a random 33% split (random_state=42).")
    st.markdown('<div class="disclaimer">Metrics describe performance on the notebook’s test split and may not generalize to other populations. This tool does not provide medical diagnosis.</div>', unsafe_allow_html=True)

