import pandas as pd

df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV8.csv')

print(df.shape)

test_df = df[['team', 'season', 'gameId', 'gameDate', 'home_team_win', 'gameNum']]
test_df = test_df[(test_df['season'] == 2023) | (test_df['season'] == 2024)]
test_df = test_df[test_df['team']=='ANA']
print(test_df.shape)
test_df.to_csv('/Users/chrisbugs/Downloads/debugSZNv1.csv')
