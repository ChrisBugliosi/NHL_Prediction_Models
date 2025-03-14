import pandas as pd
import sqlite3
import numpy as np
from os import path

# Load in CSV
DATA_DIR = '/Users/chrisbugs/Downloads'
df = pd.read_csv(path.join(DATA_DIR, 'all_teams.csv'))
print(df.shape)

# Remove playoff games
df = df[df['playoffGame'] != 1]

# Drop 2024 season (it is incomplete)
df = df[df['season'] != 2024 ]

# Remove unnecessary columns (don't need xxxxAgainst stats since we're merging home and away df's later)
df = df[['team', 'season', 'gameId', 'home_or_away', 'gameDate', 'situation', 'xGoalsPercentage', 'corsiPercentage',
         'fenwickPercentage', 'iceTime', 'xOnGoalFor', 'xGoalsFor', 'flurryAdjustedxGoalsFor', 'shotsOnGoalFor',
         'missedShotsFor', 'blockedShotAttemptsFor', 'shotAttemptsFor', 'goalsFor', 'savedShotsOnGoalFor',
         'penalityMinutesFor', 'faceOffsWonFor', 'hitsFor', 'takeawaysFor', 'giveawaysFor', 'lowDangerShotsFor',
         'mediumDangerShotsFor', 'highDangerShotsFor', 'lowDangerxGoalsFor', 'mediumDangerxGoalsFor',
         'highDangerxGoalsFor', 'lowDangerGoalsFor', 'mediumDangerGoalsFor', 'highDangerGoalsFor',
         'dZoneGiveawaysFor']]

print(df.describe())
# Split DF on home and away (for simplyfying classfying into home team win or loss)
home_df = df[df['home_or_away'] == 'HOME'].reset_index(drop=True)
away_df = df[df['home_or_away'] == 'AWAY'].reset_index(drop=True)

# Rename columns in the away DataFrame
columns_excluded = ['season', 'gameId', 'gameDate', 'home_or_away', 'situation']
away_df = away_df.rename(columns={col: f'away_{col}' for col in away_df.columns if col not in columns_excluded})

# Merge home and away DataFrames
df = pd.merge(home_df, away_df, on=['gameId', 'season', 'gameDate', 'situation'], how='inner')
df.drop(['home_or_away_x', 'home_or_away_y'], axis=1, inplace=True)

# Create a win/loss column for each game only when situation == 'all'
df = df.sort_values(by=['team', 'season', 'gameDate']).reset_index(drop=True)
# Create a temporary win/loss column for the 'all' situation
df['temp_win'] = np.where((df['situation'] == 'all') & (df['goalsFor'] > df['away_goalsFor']), 1, 0)
# Group by gameId and propagate the win/loss to all situations within the same game
df['home_team_win'] = df.groupby('gameId')['temp_win'].transform('max')

# Drop the temporary column if desired
df.drop(columns='temp_win', inplace=True)

# Create a gameNum column for each team within each season
df = df.sort_values(by=['team', 'season', 'gameDate']).reset_index(drop=True)
df['gameNum'] = df.groupby(['team', 'season'])['gameId'].transform(lambda x: pd.factorize(x)[0] + 1)
print(df[['team', 'season', 'gameId', 'situation', 'gameNum']].head())
print(df[df['season'] == 2009][['team', 'season', 'gameId', 'situation', 'gameNum']].sample(25))

# Remove situation == all (redundant as it's encapsulated in other situations)
df = df[df['situation'] != 'all']

# Check size
print(df.shape)

# Null check
print(df.isnull().sum().max())

# Save result to SQL database and CSV
save = False
if save:
    df.to_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV10.csv')
    print("DataFrame successfully saved to CSV database.")


