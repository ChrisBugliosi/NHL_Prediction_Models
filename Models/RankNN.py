import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
features_df = pd.read_csv('/Users/chrisbugs/Downloads/StandardizedHockeyFeaturesV2.csv')
total_df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')

# Preprocess Data
features_df.drop(columns=['Unnamed: 0'], inplace=True)
recent_season = total_df['season'].max()

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

X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
X_holdout = torch.tensor(holdout_features.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)
y_holdout = torch.tensor(holdout_targets.values, dtype=torch.float32)

# Neural Network Definition
class HockeyNet(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, dropout_rate):
        super(HockeyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Bayesian Optimization Function
def nn_evaluate(hidden1, hidden2, dropout_rate, lr, epochs):
    hidden1, hidden2, epochs = int(hidden1), int(hidden2), int(epochs)
    model = HockeyNet(X_train.shape[1], hidden1, hidden2, dropout_rate)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train).squeeze()
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_val_pred = (model(X_val).squeeze() > 0.5).float()
        accuracy = (y_val_pred == y_val).float().mean().item()
    return accuracy

# Perform Bayesian Optimization
optimizer = BayesianOptimization(
    f=nn_evaluate,
    pbounds={
        'hidden1': (32, 128),
        'hidden2': (16, 64),
        'dropout_rate': (0.1, 0.5),
        'lr': (0.0001, 0.01),
        'epochs': (50, 150)
    },
    random_state=42
)

print("Running Bayesian Optimization...")
optimizer.maximize(init_points=5, n_iter=15)

# Best Parameters
best_params = optimizer.max['params']
best_params['hidden1'], best_params['hidden2'], best_params['epochs'] = int(best_params['hidden1']), int(best_params['hidden2']), int(best_params['epochs'])

# Train Optimized Model
model = HockeyNet(X_train.shape[1], best_params['hidden1'], best_params['hidden2'], best_params['dropout_rate'])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])

for epoch in range(best_params['epochs']):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train).squeeze()
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

# Evaluate on Holdout Set
model.eval()
with torch.no_grad():
    y_holdout_pred = (model(X_holdout).squeeze() > 0.5).float()
    holdout_accuracy = (y_holdout_pred == y_holdout).float().mean().item()
    print(f"Holdout Accuracy: {holdout_accuracy:.4f}")

holdout_metadata = holdout_metadata.reset_index(drop=True)
holdout_metadata['score'] = y_holdout_pred.numpy()

team_rankings = holdout_metadata.groupby('team')['score'].mean().sort_values(ascending=False)

# Plot Team Rankings
plt.figure(figsize=(10, 8))
team_rankings.plot(kind='barh', color='skyblue', edgecolor='black')
plt.title('2023-24 Team Rankings Based on Optimized Neural Network Predictions')
plt.xlabel('Predicted Score')
plt.ylabel('Team')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Calculate Wins for Each Team
recent_season_data = total_df[total_df['season'] == recent_season]
unique_games = recent_season_data.drop_duplicates(subset=['gameId'])
home_wins_count = unique_games[unique_games['home_team_win'] == 1].groupby('team').size()
away_wins_count = unique_games[unique_games['home_team_win'] == 0].groupby('away_team').size()
total_wins = home_wins_count.add(away_wins_count, fill_value=0).sort_values(ascending=False)

# Final Standings
final_standings = pd.DataFrame({
    'team': total_wins.index,
    'wins': total_wins.values
}).sort_values(by='wins', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(final_standings['team'], final_standings['wins'], color='skyblue', edgecolor='black')
plt.title('Final Standings Based on Wins')
plt.xlabel('Wins')
plt.ylabel('Team')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Comparison Between Actual and Predicted Standings
nn_standings = pd.DataFrame({
    'team': team_rankings.index,
    'nn_rank': range(1, len(team_rankings) + 1)
})

actual_standings = final_standings.reset_index(drop=True)
actual_standings['actual_rank'] = range(1, len(actual_standings) + 1)

comparison = pd.merge(actual_standings, nn_standings, on='team')
comparison['rank_difference'] = abs(comparison['actual_rank'] - comparison['nn_rank'])
total_rank_difference = comparison['rank_difference'].sum()

print("Comparison of Actual vs Neural Network Predicted Standings:")
print(comparison)
print(f"\nTotal Rank Difference: {total_rank_difference}")

# Plot Rank Differences
plt.figure(figsize=(10, 8))
plt.barh(comparison['team'], comparison['rank_difference'], color='coral', edgecolor='black')
plt.title('Rank Differences Between Neural Network Predictions and Actual Standings')
plt.xlabel('Rank Difference')
plt.ylabel('Team')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
