import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# TODO: OPTIMIZE ALL THE MODELS (bayesian optimization) AND TEST ACROSS HOLDOUT SEASONS!!!!!!


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
X_train = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1)  # Add sequence dimension
X_val = torch.tensor(X_val.values, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)


# Neural Network Definition with LSTM
class HockeyLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(HockeyLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # Single output for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)  # Use the hidden state of the last LSTM cell
        x = self.fc(hidden[-1])  # Take output from the last layer
        x = self.sigmoid(x)  # Binary probability
        return x


# Initialize the LSTM model
input_size = X_train.shape[2]  # Number of features
hidden_size = 64  # LSTM hidden units
num_layers = 2  # Number of LSTM layers
model = HockeyLSTMClassifier(input_size, hidden_size, num_layers)

# Loss and optimizer
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train).squeeze()
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

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
    y_val_pred_class = (model(X_val).squeeze() > 0.5).float()
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
