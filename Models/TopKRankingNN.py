import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
features_df = pd.read_csv('/Users/chrisbugs/Downloads/StandardizedHockeyFeaturesV2.csv')
total_df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')

# Preprocess Data
features_df.drop(columns=['Unnamed: 0'], inplace=True)
recent_season = total_df['season'].max()

# Ensure data alignment
features_df['team'] = total_df['team']
features_df['gameId'] = total_df['gameId']
features_df['season'] = total_df['season']

# Split into holdout (most recent season) and training dataset
holdout_features = features_df[features_df['season'] == recent_season].drop(columns=['season', 'team', 'gameId'])
holdout_metadata = features_df[features_df['season'] == recent_season][['gameId', 'team']]
holdout_targets = total_df[total_df['season'] == recent_season]['home_team_win']

train_features = features_df[features_df['season'] != recent_season].drop(columns=['season', 'team', 'gameId'])
train_targets = total_df[total_df['season'] != recent_season]['home_team_win']

# Further split training dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, train_targets, test_size=0.2, random_state=42)

# Convert to Tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
X_holdout = torch.tensor(holdout_features.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)


# Neural Network Definition
class HockeyNet(nn.Module):
    def __init__(self, input_size):
        super(HockeyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()


# Initialize Model
input_size = X_train.shape[1]
model = HockeyNet(input_size)
criterion = nn.MarginRankingLoss(margin=1.0)  # Pairwise ranking loss
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training Loop with Pairwise Loss
def generate_pairs(X, y):
    """
    Generate pairs of features and labels for pairwise ranking.
    Returns: X1, X2, y_ranking (1 if y1 > y2 else -1)
    """
    X1, X2, y_ranking = [], [], []
    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            X1.append(X[i])
            X2.append(X[j])
            y_ranking.append(1.0 if y[i] > y[j] else -1.0)
    return torch.stack(X1), torch.stack(X2), torch.tensor(y_ranking, dtype=torch.float32)


# Training
epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Generate training pairs
    X1_train, X2_train, y_ranking_train = generate_pairs(X_train, y_train)
    scores1 = model(X1_train)
    scores2 = model(X2_train)
    loss = criterion(scores1, scores2, y_ranking_train)

    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        X1_val, X2_val, y_ranking_val = generate_pairs(X_val, y_val)
        val_scores1 = model(X1_val)
        val_scores2 = model(X2_val)
        val_loss = criterion(val_scores1, val_scores2, y_ranking_val)

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# Evaluate on Holdout Set
model.eval()
with torch.no_grad():
    y_holdout_pred = model(X_holdout)

# Align predictions with metadata
holdout_metadata = holdout_metadata.reset_index(drop=True)
holdout_metadata['score'] = y_holdout_pred.numpy()

# Generate Team Rankings
team_rankings = holdout_metadata.groupby('team')['score'].mean().sort_values(ascending=False)

# Display Top-k Teams
k = 10
print(f"Top-{k} Teams Based on Predicted Scores:")
print(team_rankings.head(k))

# Plot Top-k Rankings
plt.figure(figsize=(10, 8))
team_rankings.head(k).plot(kind='barh', color='skyblue', edgecolor='black')
plt.title(f'Top-{k} Team Rankings Based on Neural Network Predictions')
plt.xlabel('Predicted Score')
plt.ylabel('Team')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
