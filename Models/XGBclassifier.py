import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import train_test_split, cross_val_score

# Load Datasets
features_df = pd.read_csv('/Users/chrisbugs/Downloads/FeaturesHockeyDataV3.csv')
total_df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')

# Preprocess Data
features_df.drop(columns=['Unnamed: 0'], inplace=True)
recent_season = total_df['season'].max()

# Ensure Holdout Season Split
holdout_features = features_df[total_df['season'] == recent_season]
holdout_targets = total_df[total_df['season'] == recent_season]['home_team_win']


train_features = features_df[total_df['season'] != recent_season]
train_targets = total_df[total_df['season'] != recent_season]['home_team_win']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(train_features, train_targets, test_size=0.2, random_state=42)

# Define Bayesian Optimization Function for XGBoost
def xgb_evaluate(learning_rate, n_estimators, max_depth, subsample, colsample_bytree):
    params = {
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }
    model = xgb.XGBClassifier(**params)
    cv_scores = cross_val_score(model, x_train, y_train, scoring='neg_log_loss', cv=5)  # 5-Fold Cross-Validation
    return cv_scores.mean()

# Bayesian Optimization with Ranges Centered Around Defaults
optimizer = BayesianOptimization(
    f=xgb_evaluate,
    pbounds={
        'learning_rate': (0.01, 0.3),      # Refined range around default 0.1
        'n_estimators': (50, 500),         # Expanded for better exploration
        'max_depth': (1, 6),              # Default 6 in the middle
        'subsample': (0.7, 1.0),           # Avoid extremes
        'colsample_bytree': (0.7, 1.0)     # Avoid extremes
    },
    random_state=42
)

# Probe Default Hyperparameters
optimizer.probe(
    params={
        'learning_rate': 0.1,
        'n_estimators': 100,
        'max_depth': 6,
        'subsample': 1.0,
        'colsample_bytree': 1.0
    },
    lazy=True
)

print("Running Bayesian Optimization...")
optimizer.maximize(init_points=5, n_iter=15)

# Extract Best Hyperparameters
best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])

print("\nBest Parameters Found:")
print(best_params)

# Train Final Model with Best Parameters and Early Stopping
final_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss'
)
final_model.fit(
    x_train, y_train,
    eval_set=[(x_test, y_test)],
    verbose=False
)

# Evaluate on Test Set
y_test_pred = final_model.predict(x_test)
y_test_proba = final_model.predict_proba(x_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc = roc_auc_score(y_test, y_test_proba)

print("\nTest Set Results:")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test AUC: {test_auc}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Loss', 'Win']))

# Evaluate on Holdout Set
y_holdout_pred = final_model.predict(holdout_features)
y_holdout_proba = final_model.predict_proba(holdout_features)[:, 1]

holdout_accuracy = accuracy_score(holdout_targets, y_holdout_pred)
holdout_auc = roc_auc_score(holdout_targets, y_holdout_proba)

print("\nHoldout Season Results:")
print(f"Holdout Accuracy: {holdout_accuracy}")
print(f"Holdout AUC: {holdout_auc}")
print("\nClassification Report (Holdout):")
print(classification_report(holdout_targets, y_holdout_pred, target_names=['Loss', 'Win']))

# Confusion Matrix for Holdout Season
holdout_cm = confusion_matrix(holdout_targets, y_holdout_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(holdout_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.title("Holdout Season Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve for Holdout Season
holdout_fpr, holdout_tpr, _ = roc_curve(holdout_targets, y_holdout_proba)
plt.figure(figsize=(8, 6))
plt.plot(holdout_fpr, holdout_tpr, label=f"Holdout ROC Curve (AUC = {holdout_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Holdout Season)")
plt.legend()
plt.show()

# Feature Importance
feature_importances = pd.DataFrame({'feature': train_features.columns, 'importance': final_model.feature_importances_})
feature_importances.sort_values(by='importance', ascending=False, inplace=True)

num_features = len(feature_importances)
plt.figure(figsize=(12, num_features * 0.2))
plt.barh(feature_importances['feature'], feature_importances['importance'], color='skyblue', edgecolor='black')
plt.title("Feature Importance (Optimized XGB Classifier)", fontsize=16, pad=8)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.35, right=0.95)
plt.show()
