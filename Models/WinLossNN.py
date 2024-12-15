import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bayes_opt import BayesianOptimization

# Load Data
features_df = pd.read_csv('/Users/chrisbugs/Downloads/StandardizedHockeyFeaturesV2.csv')
total_df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')

# Preprocess Data
features_df.drop(columns=['Unnamed: 0'], inplace=True)
recent_season = total_df['season'].max()

# Ensure Holdout Season Split
holdout_features = features_df[total_df['season'] == recent_season]
holdout_targets = total_df[total_df['season'] == recent_season]['home_team_win']

train_features = features_df[total_df['season'] != recent_season]
train_targets = total_df[total_df['season'] != recent_season]['home_team_win']

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(train_features, train_targets, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)
X_holdout = torch.tensor(holdout_features.values, dtype=torch.float32)
y_holdout = torch.tensor(holdout_targets.values, dtype=torch.float32)


# Neural Network Definition
class HockeyClassifier(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, dropout):
        super(HockeyClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, int(hidden1))
        self.fc2 = nn.Linear(int(hidden1), int(hidden2))
        self.fc3 = nn.Linear(int(hidden2), 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


# Define the Bayesian Optimization function
def nn_evaluate(hidden1, hidden2, dropout, lr):
    input_size = X_train.shape[1]
    model = HockeyClassifier(input_size, hidden1, hidden2, dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Training loop (fewer epochs for optimization)
    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train).squeeze()
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    # Validation evaluation
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val).squeeze()
        y_val_pred_class = (y_val_pred > 0.5).float()
        val_accuracy = accuracy_score(y_val.numpy(), y_val_pred_class.numpy())
    return val_accuracy


# Bayesian Optimization
optimizer = BayesianOptimization(
    f=nn_evaluate,
    pbounds={
        'hidden1': (32, 128),
        'hidden2': (16, 64),
        'dropout': (0.1, 0.5),
        'lr': (0.0001, 0.01)
    },
    random_state=42
)

print("Running Bayesian Optimization...")
optimizer.maximize(init_points=5, n_iter=15)

# Extract Best Hyperparameters
best_params = optimizer.max['params']
print("\nBest Parameters Found:")
print(best_params)

# Train Final Model with Best Hyperparameters
final_model = HockeyClassifier(
    input_size=X_train.shape[1],
    hidden1=best_params['hidden1'],
    hidden2=best_params['hidden2'],
    dropout=best_params['dropout']
)
criterion = nn.BCELoss()
optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])

# Full Training Loop
epochs = 50
for epoch in range(epochs):
    final_model.train()
    optimizer.zero_grad()
    y_pred = final_model(X_train).squeeze()
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    final_model.eval()
    with torch.no_grad():
        y_val_pred = final_model(X_val).squeeze()
        val_loss = criterion(y_val_pred, y_val)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

# Evaluate Final Model on Holdout Set
final_model.eval()
with torch.no_grad():
    y_holdout_pred = final_model(X_holdout).squeeze()
    y_holdout_pred_class = (y_holdout_pred > 0.5).float()

# Holdout Metrics
holdout_accuracy = accuracy_score(y_holdout.numpy(), y_holdout_pred_class.numpy())
holdout_cm = confusion_matrix(y_holdout.numpy(), y_holdout_pred_class.numpy())

print("\nHoldout Season Results:")
print(f"Holdout Accuracy: {holdout_accuracy}")
print("\nClassification Report (Holdout):")
print(classification_report(y_holdout.numpy(), y_holdout_pred_class.numpy(), target_names=['Loss', 'Win']))

# Confusion Matrix for Holdout Season
plt.figure(figsize=(6, 5))
sns.heatmap(holdout_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.title("Holdout Season Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
