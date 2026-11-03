import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)



# ============================================================
# CUSTOM UI — MODERN ACADEMIC THEME
# ============================================================

st.markdown("""
<style>
    :root {
        --primary: #5146c8;
        --secondary: #3984e8;
        --ink: #18213b;
        --muted: #64708b;
        --surface: #ffffff;
        --page: #f4f6ff;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 3%, rgba(116, 103, 235, 0.12), transparent 28%),
            radial-gradient(circle at 92% 8%, rgba(56, 132, 232, 0.11), transparent 25%),
            var(--page);
        color: var(--ink);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #191d43 0%, #302f68 100%);
    }

    [data-testid="stSidebar"] * {
        color: #f7f8ff !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.42rem 0.35rem;
        border-radius: 9px;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1380px;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.025em;
    }

    h1 {
        font-weight: 800;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid #e5e8f5;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(37, 47, 94, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: #64708b;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #34328c;
        font-weight: 800;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px;
    }

    .stButton > button {
        border: 0;
        border-radius: 12px;
        padding: 0.62rem 1.2rem;
        color: white;
        font-weight: 700;
        background: linear-gradient(100deg, #5146c8, #3984e8);
        box-shadow: 0 7px 18px rgba(81, 70, 200, 0.2);
    }

    .stButton > button:hover {
        color: white;
        border: 0;
        filter: brightness(1.06);
        transform: translateY(-1px);
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid #e5e8f5;
        border-radius: 14px;
        overflow: hidden;
    }

    [data-testid="stAlert"] {
        border-radius: 13px;
    }

    hr {
        border-color: #e2e6f3;
    }

    .hero-card {
        padding: 2rem 2.1rem;
        border-radius: 0 0 24px 24px;
        margin: -0.4rem 0 1.6rem 0;
        color: white;
        background: linear-gradient(112deg, #41378f 0%, #5146c8 48%, #4388e8 100%);
        box-shadow: 0 16px 34px rgba(65, 55, 143, 0.17);
    }

    .hero-card h1, .hero-card p, .hero-card span {
        color: white !important;
    }

    .hero-kicker {
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.18em;
        opacity: 0.82;
        text-transform: uppercase;
    }

    .hero-card h1 {
        font-size: clamp(2rem, 4vw, 3rem);
        margin: 0.7rem 0;
    }

    .hero-card p {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-bottom: 0;
    }

    .info-card {
        background: white;
        border: 1px solid #e5e8f5;
        border-radius: 18px;
        padding: 1.25rem 1.4rem;
        box-shadow: 0 8px 24px rgba(37, 47, 94, 0.05);
        margin-bottom: 1rem;
    }

    .info-card h3 {
        margin-top: 0;
    }

    .muted-note {
        color: #64708b;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)



# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")


model = load_model()


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():
    return pd.read_csv("data/raw/student_performance.csv")


df = load_dataset()


# ============================================================
# FEATURE INFORMATION
# ============================================================

feature_names = [
    "attendance",
    "internal_marks",
    "study_hours",
    "previous_performance",
    "assignment_score"
]


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

try:

    coefficients = model.named_steps["model"].coef_[0]

    importance = np.abs(coefficients)

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

except Exception:

    feature_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": [0] * len(feature_names)
    })


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🎓 Student Performance")

st.sidebar.write("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Dashboard",
        "🔮 Predict Result",
        "📊 Dataset Analysis",
        "🤖 Model Performance",
        "🔍 Student Insights",
        "ℹ️ About Project"
    ]
)


# ============================================================
# PAGE 1 — DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🎓 Student Performance Prediction System")

    st.write(
        "Machine Learning based system for predicting student "
        "academic performance using academic and study-related factors."
    )

    st.divider()

    # Metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Students",
            len(df)
        )

    with col2:
        st.metric(
            "Features",
            len(feature_names)
        )

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with col4:
        st.metric(
            "Model",
            "Logistic Regression"
        )

    st.divider()

    st.subheader("📌 Project Overview")

    st.write(
        """
        This application uses machine learning to predict whether
        a student is likely to Pass or Fail based on attendance,
        internal marks, study hours, previous performance and
        assignment score.
        """
    )

    st.subheader("🔄 How the System Works")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info("1️⃣ Student Input")

    with col2:
        st.info("2️⃣ Data Preprocessing")

    with col3:
        st.info("3️⃣ ML Prediction")

    with col4:
        st.info("4️⃣ Recommendations")


# ============================================================
# PAGE 2 — PREDICT RESULT
# ============================================================

elif page == "🔮 Predict Result":

    st.title("🔮 Predict Student Result")

    st.write(
        "Enter the student's academic and study-related information."
    )

    col1, col2 = st.columns(2)

    with col1:

        attendance = st.number_input(
            "Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )

        internal_marks = st.number_input(
            "Internal Marks",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

        study_hours = st.number_input(
            "Study Hours per Day",
            min_value=0.0,
            max_value=24.0,
            value=3.0
        )

    with col2:

        previous_performance = st.number_input(
            "Previous Performance",
            min_value=0.0,
            max_value=100.0,
            value=65.0
        )

        assignment_score = st.number_input(
            "Assignment Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

    st.divider()

    if st.button(
        "🔮 Predict Result",
        width="stretch"
    ):

        student = pd.DataFrame({
            "attendance": [attendance],
            "internal_marks": [internal_marks],
            "study_hours": [study_hours],
            "previous_performance": [previous_performance],
            "assignment_score": [assignment_score]
        })

        prediction = model.predict(student)[0]

        probabilities = model.predict_proba(student)[0]

        st.subheader("Prediction")

        if prediction == "Pass":

            st.success("✅ PASS")

        else:

            st.error("❌ FAIL")

        # Probability

        st.subheader("Prediction Probability")

        probability_df = pd.DataFrame({
            "Result": model.classes_,
            "Probability": probabilities
        })

        probability_df["Probability"] = (
            probability_df["Probability"] * 100
        ).round(2)

        st.dataframe(
            probability_df,
            width="stretch"
        )

        # Feature importance

        st.subheader("📊 Most Important Factors")

        st.bar_chart(
            feature_importance.set_index("Feature")["Importance"]
        )

        # Recommendations

        st.subheader("💡 General Recommendations")

        recommendations = []

        if attendance < 75:

            recommendations.append(
                "⚠ Attendance is below the recommended level. "
                "Suggestion: Improve attendance to reduce academic risk."
            )

        if study_hours < 2:

            recommendations.append(
                "💡 Study time is relatively low. "
                "Suggestion: Consider increasing your regular study schedule."
            )

        if internal_marks < 50:

            recommendations.append(
                "⚠ Internal marks are currently low. "
                "Suggestion: Focus on upcoming internal assessments."
            )

        if previous_performance < 50:

            recommendations.append(
                "📚 Previous performance indicates areas for improvement. "
                "Suggestion: Review difficult topics and strengthen weak areas."
            )

        if assignment_score < 50:

            recommendations.append(
                "📝 Assignment score is relatively low. "
                "Suggestion: Give more attention to upcoming assignments."
            )

        if len(recommendations) == 0:

            st.success(
                "No specific improvement recommendation was triggered "
                "for the entered values."
            )

        else:

            for recommendation in recommendations:

                st.warning(recommendation)

        st.caption(
            "These are general recommendations based on the entered "
            "values. They are not guaranteed academic outcomes."
        )


# ============================================================
# PAGE 3 — DATASET ANALYSIS
# ============================================================

elif page == "📊 Dataset Analysis":

    st.title("📊 Dataset Analysis")

    st.write(
        "Explore the dataset used for training and evaluating "
        "the machine learning model."
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        width="stretch"
    )

    st.subheader("Dataset Shape")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Rows",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    st.subheader("Missing Values")

    missing_values = df.isnull().sum()

    missing_df = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values
    })

    st.dataframe(
        missing_df,
        width="stretch"
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(),
        width="stretch"
    )

    st.subheader("Feature Distributions")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    selected_column = st.selectbox(
        "Select a feature",
        numeric_columns
    )

    st.bar_chart(
        df[selected_column].value_counts().sort_index()
    )


# ============================================================
# PAGE 4 — MODEL PERFORMANCE
# ============================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    st.write(
        "Evaluation results of the trained machine learning model."
    )

    st.subheader("Model Used")

    st.info("Logistic Regression")

    st.subheader("Model Evaluation")

    st.info(
        "The original project README reports an accuracy of 79.6% "
        "for Logistic Regression. This value is shown as a reported "
        "result from the source project; it has not been independently "
        "recalculated in this application."
    )

    st.metric(
        "Reported accuracy",
        "79.6%"
    )

    st.subheader("Evaluation Metrics")

    st.write(
        "Precision, recall, and F1 score are not displayed because "
        "verified evaluation values are not currently available in "
        "the provided application files. Avoid interpreting missing "
        "metrics as zero."
    )

    st.subheader("Feature Importance")

    st.bar_chart(
        feature_importance.set_index("Feature")["Importance"]
    )


# ============================================================
# PAGE 5 — STUDENT INSIGHTS
# ============================================================

elif page == "🔍 Student Insights":

    st.title("🔍 Student Insights")

    st.write(
        "Analyze a student's academic profile and identify "
        "areas that may require attention."
    )

    col1, col2 = st.columns(2)

    with col1:

        attendance = st.slider(
            "Attendance (%)",
            0,
            100,
            75
        )

        internal_marks = st.slider(
            "Internal Marks",
            0,
            100,
            60
        )

        study_hours = st.slider(
            "Study Hours per Day",
            0.0,
            12.0,
            3.0
        )

    with col2:

        previous_performance = st.slider(
            "Previous Performance",
            0,
            100,
            65
        )

        assignment_score = st.slider(
            "Assignment Score",
            0,
            100,
            70
        )

    st.divider()

    st.subheader("Student Profile")

    profile = pd.DataFrame({
        "Factor": [
            "Attendance",
            "Internal Marks",
            "Study Hours",
            "Previous Performance",
            "Assignment Score"
        ],
        "Value": [
            attendance,
            internal_marks,
            study_hours,
            previous_performance,
            assignment_score
        ]
    })

    st.dataframe(
        profile,
        width="stretch"
    )

    st.subheader("Areas to Monitor")

    if attendance < 75:

        st.warning(
            "Attendance may require improvement."
        )

    if internal_marks < 50:

        st.warning(
            "Internal marks may require additional attention."
        )

    if study_hours < 2:

        st.info(
            "Consider maintaining a more regular study schedule."
        )

    if previous_performance < 50:

        st.warning(
            "Previous performance indicates areas that may need review."
        )

    if assignment_score < 50:

        st.warning(
            "Assignment performance may require improvement."
        )

    if (
        attendance >= 75
        and internal_marks >= 50
        and study_hours >= 2
        and previous_performance >= 50
        and assignment_score >= 50
    ):

        st.success(
            "No major areas were flagged by the current rules."
        )

    st.caption(
        "Student insights are general indicators and should not "
        "be interpreted as guaranteed academic outcomes."
    )


# ============================================================
# PAGE 6 — ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About the Project")

    st.subheader("Student Performance Prediction System")

    st.write(
        """
        This project uses machine learning to predict student
        performance based on academic and study-related factors.
        """
    )

    st.subheader("Input Features")

    st.write(
        """
        • Attendance

        • Internal Marks

        • Study Hours

        • Previous Performance

        • Assignment Score
        """
    )

    st.subheader("Machine Learning Workflow")

    st.write(
        """
        1. Dataset Collection

        2. Data Cleaning

        3. Missing Value Handling

        4. Exploratory Data Analysis

        5. Feature Selection

        6. Train-Test Split

        7. Data Preprocessing

        8. Model Training

        9. Model Evaluation

        10. Prediction

        11. Feature Importance

        12. Student Recommendations
        """
    )

    st.subheader("Technologies Used")

    st.write(
        """
        • Python

        • Pandas

        • NumPy

        • Scikit-learn

        • Matplotlib

        • Jupyter Notebook

        • Streamlit

        • Joblib
        """
    )

    st.subheader("Important Note")

    st.info(
        "This application is intended for educational and "
        "demonstration purposes. Predictions and recommendations "
        "should not be treated as guaranteed academic outcomes."
    )