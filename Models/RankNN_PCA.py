import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


# TODO: OPTIMIZE ALL THE MODELS (bayesian optimization) AND TEST ACROSS HOLDOUT SEASONS!!!!!!
# TODO: FIX THE REAL TEAM STANDINGS!!!!!!!


# Load Data
features_df = pd.read_csv('/Users/chrisbugs/Downloads/PCAFeaturesHockeyDataV2.csv')
total_df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')

# Preprocess Data
recent_season = total_df['season'].max()

# Ensure data alignment by keeping team and gameId columns in metadata
features_df['team'] = total_df['team']
features_df['gameId'] = total_df['gameId']
features_df['season'] = total_df['season']

# Split into holdout (most recent season) and training dataset
holdout_features = features_df[features_df['season'] == recent_season].drop(columns=['season', 'team', 'gameId'])
holdout_metadata = features_df[features_df['season'] == recent_season][['gameId', 'team']]  # Metadata for alignment
holdout_targets = total_df[total_df['season'] == recent_season]['home_team_win']  # Holdout targets

train_features = features_df[features_df['season'] != recent_season].drop(columns=['season', 'team', 'gameId'])
train_targets = total_df[total_df['season'] != recent_season]['home_team_win']  # Training targets

# Further split training dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, train_targets, test_size=0.2, random_state=42)

# Convert to Tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
X_holdout = torch.tensor(holdout_features.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)
y_holdout = torch.tensor(holdout_targets.values, dtype=torch.float32)

# Neural Network Definition
class HockeyNet(nn.Module):
    def __init__(self, input_size):
        super(HockeyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)  # Outputs performance score
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize Model
input_size = X_train.shape[1]
model = HockeyNet(input_size)
criterion = nn.MSELoss()  # Mean Squared Error for regression
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

# Evaluate on Holdout Set
model.eval()
with torch.no_grad():
    y_holdout_pred = model(X_holdout).squeeze()
    holdout_loss = criterion(y_holdout_pred, y_holdout)
    print(f"Holdout Loss (Most Recent Season): {holdout_loss.item():.4f}")

# Align predictions with metadata using indices
holdout_metadata = holdout_metadata.reset_index(drop=True)  # Ensure indices are aligned with predictions
holdout_metadata['score'] = y_holdout_pred.numpy()

# Debugging: Check alignment
assert len(holdout_metadata) == len(y_holdout_pred), "Mismatch between metadata and predictions!"
print(holdout_metadata.head())  # Ensure gameId, team, and score are aligned

# Aggregate predictions by team for rankings
team_rankings = (
    holdout_metadata.groupby('team')['score']
    .mean()
    .sort_values(ascending=False)  # Sort by score, not alphabetically
)

# Standardize Scores from 0 to 100
min_score = team_rankings.min()
max_score = team_rankings.max()
team_rankings = (team_rankings - min_score) / (max_score - min_score) * 100  # Scale to [0, 100]

# Display Rankings
print("Team Rankings (Most Recent Season):")
print(team_rankings)

# Plot Team Rankings
plt.figure(figsize=(10, 8))
team_rankings.plot(kind='barh', color='skyblue', edgecolor='black')
plt.title('2023-24 Team Rankings Based on Neural Network Predictions (With PCA)', fontsize=16)
plt.xlabel('Predicted Score', fontsize=14)
plt.ylabel('Team', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Determine the most recent season
recent_season = total_df['season'].max()

# Calculate wins for each team in the most recent season
# Home team wins
home_wins = total_df[(total_df['season'] == recent_season) & (total_df['home_team_win'] == 1)]
home_wins_count = home_wins.groupby('team').size()

# Away team wins
away_wins = total_df[(total_df['season'] == recent_season) & (total_df['home_team_win'] == 0)]
away_wins_count = away_wins.groupby('away_team').size()

# Combine home and away wins
total_wins = home_wins_count.add(away_wins_count, fill_value=0).sort_values(ascending=False)

# Create a final standings DataFrame
final_standings = pd.DataFrame({
    'team': total_wins.index,
    'wins': total_wins.values
}).sort_values(by='wins', ascending=False)

# Display final standings
print(final_standings)

# Plot the final standings
plt.figure(figsize=(10, 8))
plt.barh(final_standings['team'], final_standings['wins'], color='skyblue', edgecolor='black')
plt.title('Final Standings Based on Wins (Most Recent Season)', fontsize=16)
plt.xlabel('Wins', fontsize=14)
plt.ylabel('Team', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Create a DataFrame for team rankings from the neural network
nn_standings = pd.DataFrame({
    'team': team_rankings.index,
    'nn_rank': range(1, len(team_rankings) + 1)  # Rank the neural network's predicted standings
})

# Add ranks for actual standings
actual_standings = final_standings.reset_index(drop=True)
actual_standings['actual_rank'] = range(1, len(actual_standings) + 1)

# Merge neural network standings with actual standings
comparison = pd.merge(actual_standings, nn_standings, on='team')

# Calculate the rank difference for each team
comparison['rank_difference'] = abs(comparison['actual_rank'] - comparison['nn_rank'])

# Compute the total rank difference
total_rank_difference = comparison['rank_difference'].sum()

# Print results
print("Comparison of Actual vs Neural Network Predicted Standings:")
print(comparison)
print(f"\nTotal Rank Difference: {total_rank_difference}")

# Plotting Comparison
plt.figure(figsize=(10, 8))
plt.barh(comparison['team'], comparison['rank_difference'], color='coral', edgecolor='black')
plt.title('Rank Differences Between Neural Network Predictions and Actual Standings', fontsize=16)
plt.xlabel('Rank Difference', fontsize=14)
plt.ylabel('Team', fontsize=14)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()