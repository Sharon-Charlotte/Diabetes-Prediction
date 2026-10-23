# Diabetes Prediction

## Project Overview
This project explores diabetes classification using the CDC 2021 Behavioral Risk Factor Surveillance System (BRFSS) survey dataset.

It combines data preparation, exploratory data analysis, logistic regression, model evaluation, and an interactive Streamlit dashboard.

## Objectives
- Prepare and analyse survey data using Python.
- Investigate demographic, health, and lifestyle characteristics associated with reported diabetes.
- Train and evaluate a logistic regression classification model.
- Develop an interactive dashboard for demonstration.

## Technology Stack
- Python
- pandas and NumPy
- scikit-learn
- Streamlit
- Google Colab
- GitHub

## Model Performance
The evaluated test-set results at a probability threshold of 0.20 were:

| Metric | Result |
|---|---:|
| Accuracy | 79.44% |
| Precision | 35.99% |
| Recall | 65.86% |
| F1 score | 46.55% |
| ROC-AUC | 82.86% |

These metrics describe the model's performance on the evaluated test set. They do not establish clinical effectiveness.

## Data Source
CDC Behavioral Risk Factor Surveillance System (BRFSS), 2021:

https://www.cdc.gov/brfss/annual_data/annual_2021.html

The raw dataset is not included in this repository.

## Running the Dashboard
Install the required packages, ensure the trained model is available at `models/diabetes_logistic_pipeline.joblib`, and run:

`streamlit run app.py`

See the project requirements and implementation files for further instructions.

## Limitations
- The data are survey-based and may contain missing or self-reported information.
- Results depend on the selected variables, preprocessing, and classification threshold.
- The model identifies statistical patterns, not causes.
- The dashboard is an educational demonstration and is not a medical diagnostic tool.
