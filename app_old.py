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
        use_container_width=True
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
            use_container_width=True
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
        use_container_width=True
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
        use_container_width=True
    )

    st.subheader("Statistical Summary")

    st.dataframe(
        df.describe(),
        use_container_width=True
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

    st.write(
        "The model was evaluated using a separate testing dataset."
    )

    st.metric(
        "Accuracy",
        "79.6%"
    )

    st.warning(
        "Replace 79.6% with the final accuracy obtained from "
        "your actual model evaluation."
    )

    st.subheader("Evaluation Metrics")

    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ],
        "Value": [
            0.796,
            0.0,
            0.0,
            0.0
        ]
    })

    st.dataframe(
        metrics_df,
        use_container_width=True
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
        use_container_width=True
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