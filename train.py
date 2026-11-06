import os
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

os.makedirs('model', exist_ok=True)

# load breast cancer dataset
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# export test partition for evaluation
test_data = X_test.copy()
test_data['target'] = y_test
test_data.to_csv('test_data.csv', index=False)
print("Generated test_data.csv with shape:", test_data.shape)

# feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'model/scaler.pkl')

# model definitions
models = {
    'logistic_regression': LogisticRegression(),
    'decision_tree': DecisionTreeClassifier(random_state=42),
    'knn': KNeighborsClassifier(),
    'naive_bayes': GaussianNB(),
    'support_vector_machine': SVC(probability=True, random_state=42)
}

# train and serialize
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f'model/{name}.pkl')
    print(f"Saved: model/{name}.pkl")

print("Training pipeline finished successfully.")
