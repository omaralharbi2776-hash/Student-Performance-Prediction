# Student Performance Prediction — Enhanced Streamlit Interface

A Streamlit-based educational application that demonstrates how a saved machine-learning model can estimate a student's academic outcome (`Pass` / `Fail`) from five academic and study-related indicators.

> **Project attribution:** This repository is an enhanced version of the original project by [disha-birari](https://github.com/disha-birari/student-performance-prediction). The original project and its model are not claimed as entirely my own work. My contribution focuses on the interface improvements, usability, presentation, testing, and project documentation. Please retain the original license and attribution when redistributing the project.

## My contributions

- Updated the Streamlit interface with a modern, responsive visual layout.
- Organized the application into six pages: Dashboard, Predict Result, Dataset Analysis, Model Performance, Student Insights, and About Project.
- Improved the presentation of prediction results and probability values.
- Added clearer dataset summaries and interactive student-profile controls.
- Removed unverified hard-coded evaluation figures from the Model Performance page. Test metrics should only be reported after they are calculated from a documented test set.
- Tested the main application pages locally.

**Developer:** Omar Alharbi  
**Focus:** Python · Machine Learning · Streamlit · UI/UX enhancement

## Application pages

| Page | Purpose |
|---|---|
| Dashboard | Summarizes the dataset and explains the application workflow. |
| Predict Result | Accepts five student indicators and displays the saved model's prediction and probabilities. |
| Dataset Analysis | Explores sample records, missing values, descriptive statistics, and feature distributions. |
| Model Performance | Shows model information and available feature-weight information. It does not claim unverified test metrics. |
| Student Insights | Provides an interactive profile and general indicators that may need attention. |
| About Project | Describes the project purpose, input features, and technologies. |

## Input features

- `attendance` — attendance percentage.
- `internal_marks` — internal assessment marks.
- `study_hours` — study hours per day.
- `previous_performance` — previous academic performance.
- `assignment_score` — assignment score.

The application loads the saved model from `models/best_model.pkl` and the dataset from `data/raw/student_performance.csv`.

## Technology stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Run locally on Windows

1. Open the project folder in VS Code.
2. Open the VS Code terminal.
3. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

4. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation for this terminal session, run:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

5. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Start the app:

   ```powershell
   streamlit run app.py
   ```

7. Open the local address shown in the terminal (usually `http://localhost:8501`).

## Project structure

```text
student-performance-prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── raw/
│       └── student_performance.csv
├── models/
│   └── best_model.pkl
├── notebooks/
├── reports/
└── .gitignore
```

## Important limitations

- This is an educational demonstration, not a validated academic decision system.
- A prediction is an estimate from the saved model, not a guarantee of a student's future result.
- Probability values are model outputs and should not be interpreted as certainty.
- Feature weights or coefficient magnitudes do not prove causation.
- No test accuracy, precision, recall, or F1 score is claimed here unless independently calculated and documented using a defined test dataset.

## Acknowledgment and license

The original implementation and machine-learning assets are attributed to [disha-birari](https://github.com/disha-birari/student-performance-prediction). The original repository indicates an MIT license. Keep the original `LICENSE` file (including its copyright and permission notice) in this repository. If the downloaded project does not contain that file, obtain the exact license text from the original repository before publishing. This README does not replace the original license.

---

**Enhanced interface and documentation by Omar Alharbi.**
