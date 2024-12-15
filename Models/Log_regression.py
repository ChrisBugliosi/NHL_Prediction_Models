import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

"""
Logistic Regression (with Holdout Season Evaluation)
"""

# Load in data
features_df = pd.read_csv('/Users/chrisbugs/Downloads/FeaturesHockeyDataV3.csv')
total_df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')

# Preprocess Data
features_df.drop(columns=['Unnamed: 0'], inplace=True)
recent_season = total_df['season'].max()

# Ensure data alignment by keeping team and gameId columns in metadata
features_df['team'] = total_df['team']
features_df['gameId'] = total_df['gameId']
features_df['season'] = total_df['season']

# Split into holdout (most recent season) and training dataset
holdout_features = features_df[features_df['season'] == recent_season].drop(columns=['season', 'team', 'gameId'])
holdout_targets = total_df[total_df['season'] == recent_season]['home_team_win']  # Holdout targets

train_features = features_df[features_df['season'] != recent_season].drop(columns=['season', 'team', 'gameId'])
train_targets = total_df[total_df['season'] != recent_season]['home_team_win']  # Training targets

# Train-test split for non-holdout training data
x_train, x_test, y_train, y_test = train_test_split(train_features, train_targets, test_size=0.2, random_state=42)

# Create the logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)

# --- Training & Test Set Evaluation ---
# Training predictions
y_train_pred = model.predict(x_train)
y_train_proba = model.predict_proba(x_train)[:, 1]

# Test predictions
y_test_pred = model.predict(x_test)
y_test_proba = model.predict_proba(x_test)[:, 1]

# Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# AUC
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)
print(f"Train AUC: {train_auc}")
print(f"Test AUC: {test_auc}")

# Confusion Matrices
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# Plot training confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.title("Training Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot test confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curves
train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)

plt.figure(figsize=(8, 6))
plt.plot(train_fpr, train_tpr, label=f"Train ROC Curve (AUC = {train_auc:.2f})")
plt.plot(test_fpr, test_tpr, label=f"Test ROC Curve (AUC = {test_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Train & Test)")
plt.legend()
plt.show()

# --- Holdout Season Evaluation ---
y_holdout_pred = model.predict(holdout_features)
y_holdout_proba = model.predict_proba(holdout_features)[:, 1]

holdout_accuracy = accuracy_score(holdout_targets, y_holdout_pred)
holdout_auc = roc_auc_score(holdout_targets, y_holdout_proba)

print(f"Holdout Accuracy: {holdout_accuracy}")
print(f"Holdout AUC: {holdout_auc}")

# Confusion Matrix for Holdout
holdout_cm = confusion_matrix(holdout_targets, y_holdout_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(holdout_cm, annot=True, fmt='d', cmap='Oranges', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.title("Holdout Season Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Holdout ROC Curve
holdout_fpr, holdout_tpr, _ = roc_curve(holdout_targets, y_holdout_proba)

plt.figure(figsize=(8, 6))
plt.plot(holdout_fpr, holdout_tpr, label=f"Holdout ROC Curve (AUC = {holdout_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Holdout Season)")
plt.legend()
plt.show()

# --- Feature Importance ---
coefficients = pd.DataFrame({
    'Feature': x_train.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

num_features = len(coefficients)
plt.figure(figsize=(12, num_features * 0.25))
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color='skyblue', edgecolor='black')
plt.title("Feature Importance (Logistic Regression Coefficients)", fontsize=16, pad=8)
plt.xlabel("Coefficient Value", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.35, right=0.95)
plt.show()

