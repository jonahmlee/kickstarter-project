import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load prepared data
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').iloc[:, 0]
y_test = pd.read_csv('data/y_test.csv').iloc[:, 0]

# Feature selection using LASSO
print("\nPerforming LASSO feature selection...")
lasso_cv = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
selected_features = lasso_cv.coef_ != 0
X_train_selected = X_train.iloc[:, selected_features]
X_test_selected = X_test.iloc[:, selected_features]

print(f"Selected {sum(selected_features)} features out of {len(selected_features)}")

# Save selected features
joblib.dump(selected_features, 'models/selected_features.pkl')

# Define models to test with updated parameters
models = {
    'MLP': MLPClassifier(
        hidden_layer_sizes=3,
        activation='relu',
        learning_rate_init=0.0001,
        alpha=1,
        max_iter=500,
        random_state=100
    ),
    'XGBoost': xgb.XGBClassifier(
        random_state=42,
        learning_rate=0.01,
        n_estimators=1000
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        learning_rate=0.01,
        n_estimators=1000,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        C=0.1,
        max_iter=1000,
        random_state=42
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=10,
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        C=1.0,
        probability=True,
        random_state=42
    )
}

# Setup cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary to store results
results = {
    'Model': [],
    'CV Score': [],
    'Test Accuracy': [],
    'Test ROC-AUC': []
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=cv)
    cv_mean = cv_scores.mean()
    
    # Train on full training set
    model.fit(X_train_selected, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results['Model'].append(name)
    results['CV Score'].append(cv_mean)
    results['Test Accuracy'].append(accuracy)
    results['Test ROC-AUC'].append(roc_auc)
    
    # Print detailed results for each model
    print(f"\n{name} Results:")
    print(f"Cross-validation score: {cv_mean:.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, f'models/{name.lower().replace(" ", "_")}_model.pkl')

# Create results DataFrame
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

# Plot results
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='Model', y='Test Accuracy')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('models/model_comparison.png')
plt.close()

# Save results
results_df.to_csv('models/model_comparison_results.csv', index=False) 