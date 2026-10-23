# Diabetes Prediction Capstone

## Project Overview

This project explores the use of machine learning to estimate the likelihood
of a positive diabetes outcome using selected health and demographic
indicators from the CDC's 2021 Behavioral Risk Factor Surveillance System
(BRFSS).

The project includes data preparation, exploratory analysis, a
scikit-learn Logistic Regression pipeline, model evaluation, and a
Streamlit dashboard for demonstrating predictions.

**Important:** This is an educational and analytical project, not a
medical diagnostic tool. Predictions are estimates from a statistical
model and must not replace advice, screening, or diagnosis from a
qualified healthcare professional.

## Objectives

- Prepare and analyse the BRFSS dataset for a binary classification task.
- Train a Logistic Regression model using selected predictors.
- Evaluate classification performance using multiple metrics.
- Save a reusable preprocessing and prediction pipeline.
- Build an interactive dashboard using Streamlit.
- Document the workflow for reproducibility and portfolio development.

## Dataset

**Source:** U.S. Centers for Disease Control and Prevention (CDC),
Behavioral Risk Factor Surveillance System (BRFSS), 2021.

The analysis uses selected demographic, health, and lifestyle variables.
Examples include age group, sex, BMI, general health, exercise,
hypertension-related indicators, cholesterol history, and smoking status.

The raw dataset is not included in this repository. Consult the CDC's
official BRFSS documentation for the original data, variable definitions,
survey methodology, and applicable data-use terms.

Official resource:
https://www.cdc.gov/brfss/annual_data/annual_2021.html

## Methodology

1. **Data preparation:** Select relevant variables, handle missing and
   invalid responses, and prepare the target and predictor variables.
2. **Exploratory analysis:** Examine the data and relevant patterns.
3. **Preprocessing:** Apply categorical encoding and numerical
   preprocessing through a scikit-learn pipeline.
4. **Model training:** Fit a Logistic Regression classifier.
5. **Evaluation:** Assess accuracy, precision, recall, F1-score,
   ROC-AUC, PR-AUC, and the confusion matrix.
6. **Dashboard:** Use Streamlit to present model performance and allow
   users to explore example predictions.
7. **Persistence:** Save the fitted pipeline for reuse by the app.

## Model Performance

The notebook reported the following results at a classification
threshold of 0.20:

| Metric | Result |
|---|---:|
| Accuracy | 79.44% |
| Precision | 35.99% |
| Recall | 65.86% |
| F1-score | 46.55% |
| ROC-AUC | 82.86% |
| PR-AUC | 0.4227 |

The reported confusion matrix was:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 59,774 | 13,495 |
| Actual positive | 3,934 | 7,589 |

The 0.20 threshold classifies an observation as positive when its
predicted probability is at least 0.20. This threshold affects the
balance between precision and recall. These results describe the
evaluated dataset and should not be interpreted as evidence of
clinical effectiveness or performance on every population.

## Project Structure

```text
diabetes-prediction-capstone/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── models/
│   └── diabetes_logistic_pipeline.joblib
├── notebooks/
│   └── DIABETES_FINAL.ipynb
├── documentation/
├── outputs/
└── screenshots/
```

Some folders may be empty until additional documentation, outputs,
and screenshots are added.

## Installation and Usage

### 1. Clone the repository

Replace `YOUR-USERNAME` and `YOUR-REPOSITORY` with your GitHub details.

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY/diabetes-prediction-capstone
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

### 5. Explore the notebook

Open `notebooks/DIABETES_FINAL.ipynb` in Jupyter or Google Colab to
review the analysis, preprocessing, training, and evaluation steps.

## Reproducibility Notes

- The trained pipeline is stored in `models/`.
- The raw BRFSS data is intentionally excluded.
- The notebook documents the analysis and model-development workflow.
- Input feature names and categorical coding must remain consistent
  with the preprocessing used during training.
- Re-run the notebook when changing the source data, feature
  definitions, or training procedure.

## Limitations

- The model identifies statistical associations; it does not establish
  causation.
- Performance depends on the dataset, target definition, preprocessing,
  and evaluation procedure.
- The classification threshold changes the reported predictions.
- Survey-based indicators may contain reporting and measurement errors.
- Performance may differ across demographic groups and in other
  populations.
- The app is a demonstration and is not validated for clinical use.

## Tools and Technologies

- Python
- pandas and NumPy
- scikit-learn
- joblib
- Matplotlib and Seaborn
- Streamlit
- Google Colab and Jupyter Notebook
- Git and GitHub

## Responsible Use

Use the project for learning, research, and demonstration. Do not use
its output to make medical decisions, deny services, or determine an
individual's health status.

## License and Data Attribution

Check the existing repository's license and the CDC's data-use terms
before redistributing project components or data. The project does not
include the raw BRFSS dataset.
