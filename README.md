# Predictive Maintenance for Dairy Manufacturing  
### A Data-Driven Approach to Reducing Equipment Downtime

## Project Overview

Unexpected equipment failures can disrupt manufacturing operations, increase maintenance costs, and affect production reliability. Predictive maintenance uses machine data to identify conditions associated with potential failures and support maintenance decisions before breakdowns occur.

This independent portfolio project explores how machine-learning methods can support predictive maintenance in a dairy-manufacturing context. It includes exploratory data analysis, model development, evaluation, threshold selection, and an illustrative business-impact analysis.

> **Important:** This is an independent portfolio project. It is not an implementation for Arla Foods and does not use Arla’s internal data. The analysis uses the publicly available UCI AI4I 2020 Predictive Maintenance Dataset, which is synthetic.

## Business Questions

The project investigates:

- Which operating conditions are associated with machine failures?
- Can machine failures be identified using sensor measurements?
- How do different classification models perform on an imbalanced dataset?
- How does changing the classification threshold affect missed failures and false alarms?
- What could the model’s alerts imply financially under different illustrative maintenance-effectiveness assumptions?

## Dataset

**Source:** [AI4I 2020 Predictive Maintenance Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

The dataset contains 10,000 observations and 14 columns describing machine operating conditions and failure outcomes.

The analysis uses the following predictors:

- Machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

The target variable is `Machine failure`.

The dataset is highly imbalanced: 339 observations (3.39%) represent machine failures, while 9,661 represent non-failures.

### Dataset limitations

- The dataset is synthetic and represents generic industrial equipment rather than a specific dairy production line.
- Results may not generalize to real manufacturing environments without validation on operational data.
- Failure-mode indicator columns were excluded from model predictors because they directly reveal information about the target and could cause data leakage.
- The dataset supports predictive analysis but does not establish causal relationships between operating conditions and failures.

## Methodology

### 1. Data preparation

- Checked data types, missing values, and duplicate records.
- Examined failure frequency and operating-condition patterns.
- Used a stratified train-test split to preserve the failure proportion.
- Applied preprocessing within scikit-learn pipelines to reduce data leakage.

### 2. Model development

The following classification models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

Because failures are relatively rare, evaluation focused on precision, recall, F1-score, ROC-AUC, PR-AUC, and the confusion matrix rather than accuracy alone.

### 3. Threshold selection

The Random Forest classification threshold was evaluated using a validation subset of the training data. A threshold of **0.30** was selected as a practical balance between detecting failures and limiting false alarms.

The threshold is a configurable decision setting. In a real deployment, it should be selected according to operational priorities and the relative costs of missed failures and unnecessary maintenance alerts.

## Model Results

### Random Forest — Test Set

| Metric | Result |
|---|---:|
| Precision | 0.75 |
| Recall | 0.68 |
| F1-score | 0.71 |
| Accuracy | 0.98 |
| ROC-AUC | 0.965 |
| PR-AUC | 0.775 |

### Confusion Matrix

| | Predicted no failure | Predicted failure |
|---|---:|---:|
| Actual no failure | 1,917 | 15 |
| Actual failure | 22 | 46 |

At the selected threshold, the model detected 46 of 68 failures and missed 22. It generated 15 false alarms.

These results illustrate the trade-off between detecting failures and generating alerts that may not correspond to actual failures.

## Illustrative Business-Impact Analysis

A scenario-based cost analysis was developed to explore how predictive maintenance could affect costs under different assumed prevention rates.

### Illustrative assumptions

| Cost item | Assumption |
|---|---:|
| Unexpected failure | €5,000 |
| Planned maintenance intervention | €1,000 |
| False-alarm investigation | €200 |

The baseline scenario assumes all 68 test-set failures occur unexpectedly, giving an illustrative baseline cost of **€340,000**.

The analysis assumes detected failures receive planned maintenance and explores different rates at which that maintenance prevents an unexpected failure.

### Scenario Results

| Assumed prevention rate | Total scenario cost | Estimated savings | Cost reduction |
|---:|---:|---:|---:|
| 50% | €274,000 | €66,000 | 19.41% |
| 70% | €228,000 | €112,000 | 32.94% |
| 90% | €182,000 | €158,000 | 46.47% |
| 100% | €159,000 | €181,000 | 53.24% |

> **These figures are illustrative scenario calculations—not measured savings, actual Arla costs, or a forecast of real-world financial performance.** The analysis excludes implementation, monitoring, and other operational costs. The assumed costs and prevention rates would need to be validated with real manufacturing data.

## Repository Structure

```text
predictive-maintenance-dairy/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 00_data_acquisition.ipynb
├── src/
├── dashboard/
├── reports/
│   └── figures/
│       └── illustrative_savings_chart.png
├── models/
├── .gitignore
└── requirements.txt
```

## How to Reproduce the Analysis

### 1. Clone the repository

```bash
git clone https://github.com/Kester245887/predictive-maintenance-dairy.git
cd predictive-maintenance-dairy
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Obtain the dataset

Download the AI4I 2020 Predictive Maintenance Dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) and place `ai4i2020.csv` in:

```text
data/raw/ai4i2020.csv
```

### 5. Run the notebook

Open and run:

```text
notebooks/00_data_acquisition.ipynb
```

The notebook contains the data preparation, exploratory analysis, model development, evaluation, threshold selection, and illustrative business-impact analysis.

## Tools and Technologies

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- Jupyter Notebook
- Git and GitHub
- Streamlit *(planned dashboard component)*

## Future Improvements

- Develop an interactive Streamlit dashboard for predictions and business-impact scenarios.
- Add model interpretation to explain which operating conditions contribute to predictions.
- Evaluate the workflow on real, time-dependent manufacturing data.
- Incorporate operational maintenance costs and validate assumptions with domain experts.
- Establish monitoring and retraining procedures for a production setting.

## Author

**Kester Alimele**  
Mathematics | Statistical Modelling | Machine Learning | Data Analytics

This project was developed as an independent portfolio project to demonstrate the application of data analysis and machine learning to an industrial reliability problem.
