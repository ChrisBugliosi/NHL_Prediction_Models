from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# TODO: ONLY PICK 1 OF THESE MODELS TO PUT INTO FINAL NOTEBOOK (AND OPTIMIZED)

# Load standardized data
features_df = pd.read_csv('/Users/chrisbugs/Downloads/StandardizedHockeyFeaturesV2.csv')
total_df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')

# Preprocess Data
features_df.drop(columns=['Unnamed: 0'], inplace=True)
recent_season = total_df['season'].max()

# Split into holdout and training dataset
holdout_features = features_df[total_df['season'] == recent_season]
holdout_targets = total_df[total_df['season'] == recent_season]['home_team_win']

train_features = features_df[total_df['season'] != recent_season]
train_targets = total_df[total_df['season'] != recent_season]['home_team_win']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(train_features, train_targets, test_size=0.2, random_state=42)

# Function to optimize and evaluate logistic regression models
def evaluate_logistic_regression(penalty, solver, param_grid, model_name):
    print(f"\nOptimizing {model_name}...")
    model = LogisticRegression(penalty=penalty, solver=solver, max_iter=1000)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(x_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")

    # Evaluate on test set
    y_test_pred = best_model.predict(x_test)
    y_test_proba = best_model.predict_proba(x_test)[:, 1]
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    print(f"Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}")

    # Confusion matrix
    test_cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
    plt.title(f"{model_name} Test Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Evaluate on holdout set
    y_holdout_pred = best_model.predict(holdout_features)
    y_holdout_proba = best_model.predict_proba(holdout_features)[:, 1]
    holdout_accuracy = accuracy_score(holdout_targets, y_holdout_pred)
    holdout_auc = roc_auc_score(holdout_targets, y_holdout_proba)
    print(f"Holdout Accuracy: {holdout_accuracy:.4f}, Holdout AUC: {holdout_auc:.4f}")

    # Classification report
    print("\nClassification Report (Holdout):")
    print(classification_report(holdout_targets, y_holdout_pred, target_names=['Loss', 'Win']))

# Logistic Regression with L1 Regularization
param_grid_l1 = {'C': [0.01, 0.1, 1, 10, 100]}  # Regularization strength grid
evaluate_logistic_regression(penalty='l1', solver='liblinear', param_grid=param_grid_l1, model_name="Logistic Regression with L1")

# Logistic Regression with L2 Regularization
param_grid_l2 = {'C': [0.01, 0.1, 1, 10, 100]}
evaluate_logistic_regression(penalty='l2', solver='lbfgs', param_grid=param_grid_l2, model_name="Logistic Regression with L2")

# Logistic Regression with ElasticNet Regularization
param_grid_en = {
    'C': [0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0.1, 0.5, 0.7, 0.9]  # Balance between L1 and L2
}
evaluate_logistic_regression(penalty='elasticnet', solver='saga', param_grid=param_grid_en, model_name="Logistic Regression with ElasticNet")
