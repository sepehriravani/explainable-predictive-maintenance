# Explainable Predictive Maintenance

A machine learning project for predicting equipment failures using operational sensor data and an interpretable Logistic Regression model.

The project focuses on imbalanced classification, model evaluation, explainability, and error analysis.

## Project Overview

Unexpected machine failures can lead to production delays, maintenance costs, and equipment damage.

The goal of this project is to predict whether a machine will fail based on operational features such as temperature, rotational speed, torque, and tool wear.

Because failure cases are rare, the project does not rely only on accuracy. Recall, precision, F1-score, and the confusion matrix are also used to evaluate the model.

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

The dataset contains 10,000 observations.

### Input Features

* Product type
* Air temperature
* Process temperature
* Rotational speed
* Torque
* Tool wear

### Target

* `Machine failure`

  * `0`: No failure
  * `1`: Failure

Only 339 observations represent machine failures, making the dataset highly imbalanced.

## Project Workflow

1. Data exploration
2. Missing-value and duplicate checks
3. Class imbalance analysis
4. One-hot encoding
5. Train-test split
6. Feature scaling
7. Baseline evaluation
8. Logistic Regression training
9. Class-weight comparison
10. Model explainability
11. Error analysis

## Repository Structure

```text
explainable-predictive-maintenance/
│
├── README.md
├── requirements.txt
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_data_preprocessing.ipynb
    └── 03_modeling_and_evaluation.ipynb
```

## Model Results

### Baseline Model

The baseline predicts no failure for every test observation.

| Model    | Accuracy | Recall |
| -------- | -------: | -----: |
| Baseline |    96.6% |   0.0% |

Although the baseline achieves high accuracy, it fails to detect any actual machine failures.

### Logistic Regression Comparison

| Model                    | Accuracy | Precision | Recall | F1-score |
| ------------------------ | -------: | --------: | -----: | -------: |
| Without balanced weights |    96.8% |     63.6% |  10.3% |    17.7% |
| With balanced weights    |    82.4% |     14.2% |  82.4% |    24.2% |

The model without balanced class weights achieves higher accuracy and precision, but detects only about 10% of actual failures.

The balanced model detects approximately 82% of failures, although it produces more false alarms.

For predictive maintenance, missing a real failure may be more costly than generating an unnecessary warning. Therefore, the balanced Logistic Regression model was selected as the main model.

## Model Explainability

The Logistic Regression coefficients were examined to understand how each feature influenced the predictions.

The analysis showed that:

* Higher torque increased the predicted likelihood of failure.
* Greater tool wear was associated with a higher predicted likelihood of failure.
* The model used multiple operational features together when making predictions.

The coefficients represent relationships learned by the model and should not be interpreted as proof of causation.

## Error Analysis

The balanced model detected 56 of the 68 failures in the test set and missed 12 failures.

Missed failures had considerably lower average torque than correctly detected failures:

| Feature             | Missed Failures | Detected Failures |
| ------------------- | --------------: | ----------------: |
| Air temperature     |          300.99 |            300.69 |
| Process temperature |          310.36 |            310.09 |
| Rotational speed    |         1472.67 |           1481.05 |
| Torque              |           40.25 |             52.28 |
| Tool wear           |          145.50 |            155.00 |

This suggests that failures with clearer warning signals, especially high torque, were easier for the model to detect.

Failures with more normal operating values were more difficult to distinguish from healthy machines.

## Main Conclusion

This project demonstrates why accuracy alone can be misleading in an imbalanced classification problem.

The unbalanced model achieved high accuracy but missed most failures. Using balanced class weights substantially improved failure detection, but also increased the number of false alarms.

The final model prioritizes recall because detecting potential failures is the main objective of the project.

## Technologies

* Python
* Pandas
* Matplotlib
* Scikit-learn
* UCI Machine Learning Repository
* Google Colab

## Notebooks

* [01 — Data Exploration](notebooks/01_data_exploration.ipynb): Dataset inspection and exploratory analysis
* [02 — Data Preprocessing](notebooks/02_data_preprocessing.ipynb): Encoding, data splitting, and feature scaling
* [03 — Modeling and Evaluation](notebooks/03_modeling_and_evaluation.ipynb): Modeling, evaluation, explainability, and error analysis

## Limitations

* The dataset is highly imbalanced.
* Logistic Regression can mainly learn linear relationships.
* The balanced model produces many false-positive warnings.
* The results are based on one train-test split.
* Further threshold tuning and validation could improve the balance between recall and precision.

## Future Improvements

* Evaluate different classification thresholds
* Use cross-validation
* Compare Logistic Regression with tree-based models
* Evaluate Precision-Recall and ROC curves
* Reduce false alarms while maintaining high recall
