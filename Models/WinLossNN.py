import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# TODO: OPTIMIZE ALL THE MODELS (bayesian optimization)!!!!!!


# Load Data
features_df = pd.read_csv('/Users/chrisbugs/Downloads/StandardizedHockeyFeaturesV2.csv')
total_df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')

# Preprocess Data
features_df.drop(columns=['Unnamed: 0'], inplace=True)

# Separate features and target variable
X = features_df
y = total_df['home_team_win']

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)

# Neural Network Definition
class HockeyClassifier(nn.Module):
    def __init__(self, input_size):
        super(HockeyClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Single output for binary classification
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))  # Output a probability between 0 and 1
        return x


# Initialize the model
input_size = X_train.shape[1]
model = HockeyClassifier(input_size)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train).squeeze()  # Get predictions
    loss = criterion(y_pred, y_train)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    # Validation loss
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val).squeeze()
        val_loss = criterion(y_val_pred, y_val)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")

# Evaluate the Model
model.eval()
with torch.no_grad():
    y_val_pred_class = (model(X_val).squeeze() > 0.5).float()  # Convert probabilities to binary predictions
    y_train_pred_class = (model(X_train).squeeze() > 0.5).float()

# Classification Metrics
print("Validation Metrics:")
print(classification_report(y_val.numpy(), y_val_pred_class.numpy(), target_names=['Loss', 'Win']))

# Confusion Matrix
val_cm = confusion_matrix(y_val.numpy(), y_val_pred_class.numpy())
plt.figure(figsize=(6, 5))
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.title("Validation Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
