import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Student Performance | AI Insights",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Modern blue-purple visual theme
# ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background:
      radial-gradient(circle at 10% 0%, rgba(119, 92, 255, .10), transparent 28%),
      radial-gradient(circle at 90% 10%, rgba(42, 150, 255, .10), transparent 25%),
      #f6f7fc;
    color: #18223b;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #171b3d 0%, #25265a 55%, #35317a 100%);
}
[data-testid="stSidebar"] * { color: #f5f6ff !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 8px 10px; border-radius: 10px;
}
h1 { color: #1d2450; font-weight: 800 !important; letter-spacing: -1.1px; }
h2, h3 { color: #252b58; font-weight: 700 !important; }
p, label { color: #53607a; }
.hero {
    padding: 28px 30px; border-radius: 22px; color: white;
    background: linear-gradient(115deg, #34328a 0%, #5d50c9 55%, #398be8 100%);
    box-shadow: 0 14px 35px rgba(67, 70, 160, .18);
    margin: 4px 0 22px 0;
}
.hero h1 { color: white !important; margin: 0 0 8px 0; }
.hero p { color: rgba(255,255,255,.86) !important; margin: 0; font-size: 1rem; }
.eyebrow {
    color: #d9dcff; text-transform: uppercase; letter-spacing: 2px;
    font-size: .72rem; font-weight: 700; margin-bottom: 9px;
}
.section-card {
    background: rgba(255,255,255,.88); border: 1px solid #e7e9f5;
    border-radius: 18px; padding: 19px 21px; margin: 8px 0 16px 0;
    box-shadow: 0 7px 24px rgba(31, 42, 87, .045);
}
.small-label { color: #737d99; font-size: .78rem; font-weight: 600; }
div[data-testid="stMetric"] {
    background: white; border: 1px solid #e8eaf5; padding: 15px 17px;
    border-radius: 16px; box-shadow: 0 6px 20px rgba(31,42,87,.045);
}
div[data-testid="stMetric"] label { color: #68738f !important; }
div[data-testid="stMetricValue"] { color: #292c68 !important; font-weight: 800; }
.stButton > button {
    border: 0; border-radius: 12px; color: white;
    background: linear-gradient(100deg, #5147bd, #398be8);
    font-weight: 700; padding: .65rem 1rem;
    box-shadow: 0 7px 18px rgba(76, 81, 190, .18);
}
.stButton > button:hover { color: white; border: 0; filter: brightness(1.06); }
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    border: 1px solid #e7e9f5; border-radius: 13px; overflow: hidden;
}
hr { border-color: #e5e8f3; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load model and dataset
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")

@st.cache_data
def load_dataset():
    return pd.read_csv("data/raw/student_performance.csv")

try:
    model = load_model()
    df = load_dataset()
except Exception as error:
    st.error("The model or dataset could not be loaded. Check that the app is running from the project folder and that the models/ and data/ folders are present.")
    st.exception(error)
    st.stop()

feature_names = [
    "attendance",
    "internal_marks",
    "study_hours",
    "previous_performance",
    "assignment_score",
]

# ------------------------------------------------------------
# Feature importance (when supported by the saved pipeline)
# ------------------------------------------------------------
try:
    coefficients = model.named_steps["model"].coef_[0]
    importance = np.abs(coefficients)
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance,
    }).sort_values("Importance", ascending=False)
except Exception:
    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": [0] * len(feature_names),
    })

# ------------------------------------------------------------
# Shared page header
# ------------------------------------------------------------
def page_header(kicker, title, description):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{kicker}</div>'
        f'<h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Student AI")
    st.caption("ACADEMIC PERFORMANCE PLATFORM")
    st.markdown("---")
    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Dashboard",
            "🔮 Predict Result",
            "📊 Dataset Analysis",
            "🤖 Model Performance",
            "🔍 Student Insights",
            "ℹ️ About Project",
        ],
        label_visibility="visible",
    )
    st.markdown("---")
    st.caption("Machine Learning • Streamlit")
    st.caption("Educational demonstration")

# ------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------
if page == "🏠 Dashboard":
    page_header(
        "STUDENT PERFORMANCE ANALYTICS",
        "Understand performance. Support progress.",
        "A machine-learning application that estimates student pass/fail outcomes from academic and study-related indicators.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students in dataset", f"{len(df):,}")
    c2.metric("Input indicators", len(feature_names))
    c3.metric("Missing values", f"{int(df.isnull().sum().sum()):,}")
    c4.metric("Prediction model", "Logistic Regression")

    st.markdown("### Project overview")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("""
        <div class="section-card">
          <h3>🎯 What does this app do?</h3>
          <p>It uses five student-related indicators to estimate whether a student is likely to pass or fail.</p>
          <p><b>Inputs:</b> attendance, internal marks, study hours, previous performance, and assignment score.</p>
          <p><b>Outputs:</b> predicted class, model probability (when available), and general improvement suggestions.</p>
        </div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown("""
        <div class="section-card">
          <h3>✨ Explore the platform</h3>
          <p>🔮 <b>Predict Result</b> — enter a student profile.</p>
          <p>📊 <b>Dataset Analysis</b> — inspect the data.</p>
          <p>🤖 <b>Model Performance</b> — review model information.</p>
          <p>🔍 <b>Student Insights</b> — identify indicators to monitor.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### How it works")
    steps = st.columns(4)
    for col, number, label, detail in zip(
        steps,
        ["01", "02", "03", "04"],
        ["Student input", "Model processing", "Prediction", "Guidance"],
        ["Enter five indicators", "Saved ML pipeline", "Pass / Fail estimate", "General suggestions"],
    ):
        with col:
            st.markdown(
                f'<div class="section-card"><div class="small-label">STEP {number}</div>'
                f'<h3>{label}</h3><p>{detail}</p></div>',
                unsafe_allow_html=True,
            )

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------
elif page == "🔮 Predict Result":
    page_header(
        "PREDICTION WORKSPACE",
        "Predict student outcome",
        "Enter the student's information below to generate a model-based estimate.",
    )

    with st.form("prediction_form"):
        st.markdown("### Academic profile")
        col1, col2 = st.columns(2)
        with col1:
            attendance = st.number_input("Attendance (%)", 0.0, 100.0, 75.0, 1.0)
            internal_marks = st.number_input("Internal marks", 0.0, 100.0, 60.0, 1.0)
            study_hours = st.number_input("Study hours per day", 0.0, 24.0, 3.0, 0.5)
        with col2:
            previous_performance = st.number_input("Previous performance", 0.0, 100.0, 65.0, 1.0)
            assignment_score = st.number_input("Assignment score", 0.0, 100.0, 70.0, 1.0)
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("✨ Generate prediction", use_container_width=True)

    if submitted:
        student = pd.DataFrame({
            "attendance": [attendance],
            "internal_marks": [internal_marks],
            "study_hours": [study_hours],
            "previous_performance": [previous_performance],
            "assignment_score": [assignment_score],
        })

        try:
            prediction = model.predict(student)[0]
            probabilities = model.predict_proba(student)[0]

            st.markdown("### Prediction result")
            result_col, confidence_col = st.columns(2)
            with result_col:
                if str(prediction).lower() == "pass":
                    st.success("## ✅ PASS")
                else:
                    st.error("## ⚠️ FAIL")
                st.caption("Predicted class from the saved machine-learning model.")
            with confidence_col:
                st.markdown('<div class="section-card"><div class="small-label">MODEL PROBABILITIES</div>', unsafe_allow_html=True)
                probability_df = pd.DataFrame({
                    "Result": model.classes_,
                    "Probability (%)": (probabilities * 100).round(2),
                })
                st.dataframe(probability_df, use_container_width=True, hide_index=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### Factors considered")
            if feature_importance["Importance"].sum() > 0:
                st.bar_chart(feature_importance.set_index("Feature")["Importance"])
            else:
                st.info("Feature-importance coefficients are not available for this saved model.")

            st.markdown("### Suggested areas to review")
            recommendations = []
            if attendance < 75:
                recommendations.append("Attendance is below 75%. Consider improving class attendance.")
            if study_hours < 2:
                recommendations.append("Study time is below 2 hours per day. A consistent study routine may help.")
            if internal_marks < 50:
                recommendations.append("Internal marks are below 50. Review topics and prepare for assessments.")
            if previous_performance < 50:
                recommendations.append("Previous performance is below 50. Identify difficult topics and revisit them.")
            if assignment_score < 50:
                recommendations.append("Assignment score is below 50. Give extra attention to upcoming assignments.")

            if recommendations:
                for item in recommendations:
                    st.warning(item)
            else:
                st.success("No review suggestions were triggered by the current guidance rules.")

            st.caption("Predictions and suggestions are educational estimates, not guaranteed academic outcomes.")
        except Exception as error:
            st.error("The prediction could not be completed. Please check the model and input feature configuration.")
            st.exception(error)

# ------------------------------------------------------------
# Dataset analysis
# ------------------------------------------------------------
elif page == "📊 Dataset Analysis":
    page_header(
        "DATA EXPLORATION",
        "Explore the student dataset",
        "Review sample records, dataset dimensions, missing values, and numeric feature distributions.",
    )

    st.markdown("### Dataset snapshot")
    a, b, c = st.columns(3)
    a.metric("Records", f"{df.shape[0]:,}")
    b.metric("Columns", df.shape[1])
    c.metric("Missing values", int(df.isnull().sum().sum()))

    st.markdown("### Sample records")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.markdown("### Missing values by column")
        missing_df = df.isnull().sum().rename("Missing values").reset_index()
        missing_df.columns = ["Column", "Missing values"]
        st.dataframe(missing_df, use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Statistical summary")
        st.dataframe(df.describe().T, use_container_width=True)

    st.markdown("### Feature distribution")
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_columns:
        selected_column = st.selectbox("Choose a numeric feature", numeric_columns)
        st.bar_chart(df[selected_column].value_counts().sort_index())
    else:
        st.info("No numeric columns were found in the dataset.")

# ------------------------------------------------------------
# Model performance
# ------------------------------------------------------------
elif page == "🤖 Model Performance":
    page_header(
        "MODEL INFORMATION",
        "Understand the prediction model",
        "A transparent overview of the model currently loaded by this application.",
    )

    st.markdown("### Active model")
    st.info("Logistic Regression — loaded from `models/best_model.pkl`.")

    st.markdown("### Model feature weights")
    if feature_importance["Importance"].sum() > 0:
        st.write("The chart shows the absolute coefficient magnitudes exposed by the saved model pipeline. Larger values indicate larger coefficient magnitudes, not proof of causation.")
        st.bar_chart(feature_importance.set_index("Feature")["Importance"])
    else:
        st.info("Coefficient-based feature importance is not available for this model.")

    st.markdown("### Evaluation metrics")
    st.warning(
        "Verified test-set metrics are not calculated by this app. "
        "The earlier hard-coded accuracy and zero-valued metrics have been removed "
        "so the interface does not present unverified values as measured results."
    )
    st.markdown("""
    <div class="section-card">
      <h3>What should be reported?</h3>
      <p>Evaluate the saved model on a held-out test set and report accuracy, precision, recall, F1-score, and a confusion matrix. Only add numerical results after reproducing the evaluation from the project data and code.</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------
# Student insights
# ------------------------------------------------------------
elif page == "🔍 Student Insights":
    page_header(
        "STUDENT PROFILE",
        "Review learning indicators",
        "Adjust the indicators to see which areas the simple guidance rules flag for review.",
    )

    col1, col2 = st.columns(2)
    with col1:
        attendance = st.slider("Attendance (%)", 0, 100, 75)
        internal_marks = st.slider("Internal marks", 0, 100, 60)
        study_hours = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
    with col2:
        previous_performance = st.slider("Previous performance", 0, 100, 65)
        assignment_score = st.slider("Assignment score", 0, 100, 70)

    profile = pd.DataFrame({
        "Indicator": ["Attendance", "Internal marks", "Study hours/day", "Previous performance", "Assignment score"],
        "Value": [attendance, internal_marks, study_hours, previous_performance, assignment_score],
    })
    st.markdown("### Profile summary")
    st.dataframe(profile, use_container_width=True, hide_index=True)

    st.markdown("### Areas to monitor")
    flags = []
    if attendance < 75:
        flags.append("Attendance is below 75%.")
    if internal_marks < 50:
        flags.append("Internal marks are below 50.")
    if study_hours < 2:
        flags.append("Study time is below 2 hours per day.")
    if previous_performance < 50:
        flags.append("Previous performance is below 50.")
    if assignment_score < 50:
        flags.append("Assignment score is below 50.")

    if flags:
        for flag in flags:
            st.warning(flag)
    else:
        st.success("No areas were flagged by the current guidance rules.")

    st.caption("These are simple rule-based indicators, not a diagnosis or a guaranteed prediction.")

# ------------------------------------------------------------
# About
# ------------------------------------------------------------
elif page == "ℹ️ About Project":
    page_header(
        "PROJECT INFORMATION",
        "About this application",
        "A Streamlit interface for exploring a student-performance prediction project.",
    )

    st.markdown("""
    <div class="section-card">
      <h3>🎓 Project purpose</h3>
      <p>This application demonstrates how a saved machine-learning model can be used to estimate student pass/fail outcomes from academic and study-related indicators.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Input features")
    feature_cols = st.columns(5)
    for col, feature in zip(feature_cols, ["Attendance", "Internal marks", "Study hours", "Previous performance", "Assignment score"]):
        with col:
            st.markdown(f'<div class="section-card"><h3>📌</h3><p><b>{feature}</b></p></div>', unsafe_allow_html=True)

    st.markdown("### Technologies")
    st.write("Python · Pandas · NumPy · Scikit-learn · Streamlit · Joblib")

    st.markdown("### Important note")
    st.info("This application is intended for educational and demonstration purposes. Predictions and recommendations should not be treated as guaranteed academic outcomes.")
    st.caption("If you build on the original repository, retain its license and attribution notices and clearly describe your own modifications.")
