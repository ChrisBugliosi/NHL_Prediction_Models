import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import train_test_split


# TODO: OPTIMIZE ALL THE MODELS (bayesian optimization)!!!!!!


"""
XGB Classifier w/ default data
"""
# Load datasets
features_df = pd.read_csv('/Users/chrisbugs/Downloads/FeaturesHockeyDataV3.csv')
total_df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')

# Prepare features and labels
features_df.drop(columns=['Unnamed: 0'], inplace=True)
x_features = features_df  # Features
y_classes = total_df['home_team_win']  # Target variable

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_features, y_classes, test_size=0.2, random_state=42)

# Initialize and train the XGB model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')  # Disable warning and set evaluation metric
model.fit(x_train, y_train)

# Predictions
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Classification report for additional insights
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['Loss', 'Win']))

# Probabilities for ROC/AUC
y_train_proba = model.predict_proba(x_train)[:, 1]
y_test_proba = model.predict_proba(x_test)[:, 1]

# AUC scores
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)
print(f"Train AUC: {train_auc}")
print(f"Test AUC: {test_auc}")

# Confusion matrices
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# Training Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.title("Training Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Test Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Compute ROC Curve
train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)

# ROC Curve Plot
plt.figure(figsize=(8, 6))
plt.plot(train_fpr, train_tpr, label=f"Train ROC Curve (AUC = {train_auc:.2f})")
plt.plot(test_fpr, test_tpr, label=f"Test ROC Curve (AUC = {test_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Feature Importance Plot
feature_importances = pd.DataFrame({'feature': x_features.columns, 'importance': model.feature_importances_})
feature_importances.sort_values(by='importance', ascending=False, inplace=True)

plt.figure(figsize=(12, 6))
plt.barh(feature_importances['feature'], feature_importances['importance'], color='skyblue', edgecolor='black')
plt.title("Feature Importance (XGB Classifier)", fontsize=16, pad=8)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.35, right=0.95)
plt.show()

"""
XGB Classifier w/ PCA data
"""
# Load datasets
features_df = pd.read_csv('/Users/chrisbugs/Downloads/PCAFeaturesHockeyDataV2.csv')
total_df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')

# Prepare features and labels
x_features = features_df  # Features
y_classes = total_df['home_team_win']  # Target variable

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_features, y_classes, test_size=0.2, random_state=42)

# Initialize and train the XGB model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')  # Disable warning and set evaluation metric
model.fit(x_train, y_train)

# Predictions
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Accuracy scores
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Classification report for additional insights
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred, target_names=['Loss', 'Win']))

# Probabilities for ROC/AUC
y_train_proba = model.predict_proba(x_train)[:, 1]
y_test_proba = model.predict_proba(x_test)[:, 1]

# AUC scores
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)
print(f"Train AUC: {train_auc}")
print(f"Test AUC: {test_auc}")

# Confusion matrices
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# Training Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.title("Training Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Test Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Compute ROC Curve
train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)

# ROC Curve Plot
plt.figure(figsize=(8, 6))
plt.plot(train_fpr, train_tpr, label=f"Train ROC Curve (AUC = {train_auc:.2f})")
plt.plot(test_fpr, test_tpr, label=f"Test ROC Curve (AUC = {test_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Feature Importance Plot
feature_importances = pd.DataFrame({'feature': x_features.columns, 'importance': model.feature_importances_})
feature_importances.sort_values(by='importance', ascending=False, inplace=True)

plt.figure(figsize=(12, 6))
plt.barh(feature_importances['feature'], feature_importances['importance'], color='skyblue', edgecolor='black')
plt.title("Feature Importance (XGB Classifier)", fontsize=16, pad=8)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.35, right=0.95)
plt.show()