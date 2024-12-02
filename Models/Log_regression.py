import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


"""
Logistic Regression
"""
# Load in data
features_df = pd.read_csv('/Users/chrisbugs/Downloads/FeaturesHockeyDataV3.csv')
total_df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')


features_df.drop(columns=['Unnamed: 0'], inplace=True)
# separate the features from the y variable
x_features = features_df
y_classes = total_df['home_team_win']

# break them into test train split
x_train, x_test, y_train, y_test = train_test_split(x_features, y_classes, test_size=0.2, random_state=42)

# create the logistic regression model (note that max iter is number of iterations to occur during optimization process)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# make predictions on the test set
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

# accuracy the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

# probability of the "win" class
y_train_proba = model.predict_proba(x_train)[:, 1]
y_test_proba = model.predict_proba(x_test)[:, 1]

# get the AUC scores
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)
print(f"Train AUC: {train_auc}")
print(f"Test AUC: {test_auc}")

# confusion matrix
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

# plot train confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(train_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.title("Training Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# plot test confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.title("Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# compute ROC curve for train and test
train_fpr, train_tpr, _ = roc_curve(y_train, y_train_proba)
test_fpr, test_tpr, _ = roc_curve(y_test, y_test_proba)

# plot ROC curves
plt.figure(figsize=(8, 6))
plt.plot(train_fpr, train_tpr, label=f"Train ROC Curve (AUC = {train_auc:.2f})")
plt.plot(test_fpr, test_tpr, label=f"Test ROC Curve (AUC = {test_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guessing")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# extract coefficients
coefficients = pd.DataFrame({
    'Feature': x_features.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

# Plot feature importance
num_features = len(coefficients)
plt.figure(figsize=(12, num_features * 0.25))
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color='skyblue', edgecolor='black')
plt.title("Feature Importance (Logistic Regression Coefficients)", fontsize=16, pad=8)  # Adjust padding for title
plt.xlabel("Coefficient Value", fontsize=14)
plt.ylabel("Feature", fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.35, right=0.95)
plt.show()