from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# TODO: OPTIMIZE ALL THE MODELS (bayesian optimization) AND TEST ACROSS HOLDOUT SEASONS!!!!!!


"""
Logistic Regression w/ L1 & standardized data
"""
# Load standardized data
features_df = pd.read_csv('/Users/chrisbugs/Downloads/StandardizedHockeyFeaturesV2.csv')
total_df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')

features_df.drop(columns=['Unnamed: 0'], inplace=True)
x_features = features_df
y_classes = total_df['home_team_win']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_features, y_classes, test_size=0.2, random_state=42)

# Create the Logistic Regression model with L1 regularization
model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000, C=1.0)  # L1 regularization
model.fit(x_train, y_train)

# Make predictions
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Probability predictions for AUC
y_train_proba = model.predict_proba(x_train)[:, 1]
y_test_proba = model.predict_proba(x_test)[:, 1]

# Calculate AUC scores
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)
print(f"Train AUC: {train_auc}")
print(f"Test AUC: {test_auc}")

# Confusion matrices
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

# ROC curve for train and test
train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(train_fpr, train_tpr, label=f"Train ROC Curve (AUC = {train_auc:.2f})")
plt.plot(test_fpr, test_tpr, label=f"Test ROC Curve (AUC = {test_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Extract and visualize coefficients
coefficients = pd.DataFrame({
    'Feature': x_features.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

# Plot feature importance
num_features = len(coefficients)
plt.figure(figsize=(12, num_features * 0.25))
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color='skyblue', edgecolor='black')
plt.title("Feature Importance (Logistic Regression Coefficients with L1)", fontsize=16, pad=8)
plt.xlabel("Coefficient Value", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.35, right=0.95)
plt.show()

"""
Logistic Regression w/ L2 & standardized data
"""
# Load standardized data
features_df = pd.read_csv('/Users/chrisbugs/Downloads/StandardizedHockeyFeaturesV2.csv')
total_df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')

features_df.drop(columns=['Unnamed: 0'], inplace=True)
x_features = features_df
y_classes = total_df['home_team_win']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_features, y_classes, test_size=0.2, random_state=42)

# Create the Logistic Regression model with L2 regularization
model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000, C=1.0)  # L2 regularization
model.fit(x_train, y_train)

# Make predictions
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Probability predictions for AUC
y_train_proba = model.predict_proba(x_train)[:, 1]
y_test_proba = model.predict_proba(x_test)[:, 1]

# Calculate AUC scores
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)
print(f"Train AUC: {train_auc}")
print(f"Test AUC: {test_auc}")

# Confusion matrices
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

# ROC curve for train and test
train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(train_fpr, train_tpr, label=f"Train ROC Curve (AUC = {train_auc:.2f})")
plt.plot(test_fpr, test_tpr, label=f"Test ROC Curve (AUC = {test_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Extract and visualize coefficients
coefficients = pd.DataFrame({
    'Feature': x_features.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

# Plot feature importance
num_features = len(coefficients)
plt.figure(figsize=(12, num_features * 0.25))
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color='skyblue', edgecolor='black')
plt.title("Feature Importance (Logistic Regression Coefficients with L2)", fontsize=16, pad=8)
plt.xlabel("Coefficient Value", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.35, right=0.95)
plt.show()


"""
Logistic Regression w/ L1 and L2 & standardized data
"""
# Load standardized data
features_df = pd.read_csv('/Users/chrisbugs/Downloads/StandardizedHockeyFeaturesV2.csv')
total_df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')

features_df.drop(columns=['Unnamed: 0'], inplace=True)
x_features = features_df
y_classes = total_df['home_team_win']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_features, y_classes, test_size=0.2, random_state=42)

# Logistic Regression with ElasticNet (L1 and L2 penalties combined)
model = LogisticRegression(
    penalty='elasticnet',  # Specify ElasticNet
    solver='saga',         # Required solver for ElasticNet
    max_iter=1000,         # Number of iterations
    l1_ratio=0.5,          # Balance between L1 and L2 (0.5 means equal contribution)
    C=1.0                  # Regularization strength
)
model.fit(x_train, y_train)

# Make predictions
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# Probability predictions for AUC
y_train_proba = model.predict_proba(x_train)[:, 1]
y_test_proba = model.predict_proba(x_test)[:, 1]

# Calculate AUC scores
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)
print(f"Train AUC: {train_auc}")
print(f"Test AUC: {test_auc}")

# Confusion matrices
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

# ROC curve for train and test
train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)

# Plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(train_fpr, train_tpr, label=f"Train ROC Curve (AUC = {train_auc:.2f})")
plt.plot(test_fpr, test_tpr, label=f"Test ROC Curve (AUC = {test_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Extract and visualize coefficients
coefficients = pd.DataFrame({
    'Feature': x_features.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

# Plot feature importance
num_features = len(coefficients)
plt.figure(figsize=(12, num_features * 0.25))
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color='skyblue', edgecolor='black')
plt.title("Feature Importance (Logistic Regression Coefficients with ElasticNet)", fontsize=16, pad=8)
plt.xlabel("Coefficient Value", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.35, right=0.95)
plt.show()