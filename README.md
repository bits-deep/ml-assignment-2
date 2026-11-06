# Breast Cancer Classification Model Evaluation

## a. Problem statement
The objective of this project is to classify breast cancer biopsy samples as either malignant (0) or benign (1) based on cell nucleus characteristics computed from digitized fine needle aspirate (FNA) images. We train and evaluate five standard classification algorithms to identify the optimal predictor for clinical screening.

## b. Dataset description
- **Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source**: UCI Machine Learning Repository / Scikit-Learn
- **Total Instances**: 569 (Satisfies the minimum 500 requirement)
- **Feature Count**: 30 numeric features (Satisfies the minimum 12 requirement)
- **Target Classes**: Binary classification (0 = Malignant, 1 = Benign)
- **Split Ratio**: 80% Training (455 samples) and 20% Testing (114 samples saved as `test_data.csv`)

## c. Github Repository Link
[Insert your GitHub Repository Link here]

## d. Models used:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9737 | 0.9974 | 0.9722 | 0.9859 | 0.9790 | 0.9439 |
| Decision Tree | 0.9474 | 0.9440 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| KNN | 0.9474 | 0.9820 | 0.9577 | 0.9577 | 0.9577 | 0.8880 |
| Naive Bayes | 0.9649 | 0.9974 | 0.9589 | 0.9859 | 0.9722 | 0.9253 |
| Support Vector Machine | 0.9825 | 0.9974 | 0.9726 | 1.0000 | 0.9861 | 0.9630 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Showed strong linear separability on the standardized features, achieving 0.9737 accuracy and an AUC of 0.9974 with balanced precision and recall. |
| Decision Tree | Achieved 0.9474 accuracy. Being an unpruned tree, it showed slightly lower generalization compared to linear boundary models due to localized splits. |
| KNN | Reached 0.9474 accuracy and 0.9820 AUC. Standardizing feature scales was critical for distance metrics to avoid dominance by large variance features. |
| Naive Bayes | Handled the diagnostic attributes well with 0.9649 accuracy and 0.9974 AUC, despite the independence assumption among correlated geometric measurements. |
| Support Vector Machine | Delivered the top performance with 0.9825 accuracy, perfect recall (1.0000 on benign cases), and an MCC of 0.9630, separating high-dimensional margins reliably. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)** achieved the best overall score across Accuracy (0.9825), F1 (0.9861), and MCC (0.9630). Logistic Regression is a close second with simpler linear interpretability. |
