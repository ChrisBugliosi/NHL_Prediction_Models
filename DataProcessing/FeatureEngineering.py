import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from os import path

# https://docs.google.com/document/d/1Ao8ajyB9srdtSy5bvKV_IRaFoBfm1KU06zL81__sZfU/edit?tab=t.0#heading=h.35nkun2


# Load in CSV
df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')

df.drop(columns='Unnamed: 0', inplace=True)

# Sort DF
df = df.sort_values(by=['team', 'season', 'gameNum']).reset_index(drop=True)

# Columns excluded
exclude_cols = ['team', 'situation', 'gameId', 'gameDate', 'gameNum', 'season', 'away_team', 'home_team_win']

# Columns included
rolling_cols = [col for col in df.columns if col not in exclude_cols]

# New goal diff feature
df['goalDiff'] = df['goalsFor'] - df['away_goalsFor']

# Rolling average Calc
run_calc = False
if run_calc:
    # Calculate rolling averages for last 10
    for col in rolling_cols:
        df[f'{col}_rolling_avg_10'] = df.groupby(['team', 'season'])[col].shift(1).rolling(window=10, min_periods=1).mean()

    # For the first value, do this backwards
    for col in rolling_cols:
        forward_avg = df.groupby(['team', 'season'])[col].transform(lambda x: x.shift(-1).rolling(window=10, min_periods=1).mean())
        first_row_mask = df.groupby(['team', 'season']).cumcount() == 0
        df.loc[first_row_mask, f'{col}_rolling_avg_10'] = forward_avg[first_row_mask]

    # Features only DF
    feature_cols = [f'{col}_rolling_avg_10' for col in rolling_cols]
    feautres_df = df[feature_cols]

# Null check
print(df.isnull().sum().max())

# Save result to SQL database and CSV
save = False
if save:
    feautres_df.to_csv('/Users/chrisbugs/Downloads/FeaturesHockeyDataV4.csv')
    print("DataFrame successfully saved to CSV database.")


# Standardizing the data (for models that prefer standardized data)
standardize = False
if standardize:
    # Re-load in CSV to avoid re-doing calculations
    features_df = pd.read_csv('/Users/chrisbugs/Downloads/FeaturesHockeyDataV3.csv')

    # Standardized features
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features_df)
    standardized_features_df = pd.DataFrame(standardized_features, columns=features_df.columns)
    standardized_features_df.drop(columns='Unnamed: 0', inplace=True)

save = False
if save:
    standardized_features_df.to_csv('/Users/chrisbugs/Downloads/StandardizedHockeyFeaturesV3.csv')
    print("DataFrame successfully saved to CSV database.")

# Normalizing the data (for models that prefer normalized data)
normalize = False
if normalize:
    # Re-load in CSV to avoid re-doing calculations
    features_df = pd.read_csv('/Users/chrisbugs/Downloads/FeaturesHockeyDataV3.csv')

    # Normalize features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features_df)
    normalized_features_df = pd.DataFrame(normalized_features, columns=features_df.columns)
    normalized_features_df.drop(columns='Unnamed: 0', inplace=True)

save = False
if save:
    normalized_features_df.to_csv('/Users/chrisbugs/Downloads/NormalizedHockeyFeaturesV3.csv')
    print("DataFrame successfully saved to CSV database.")

# PCA to reduce multicollinearity
pca_run = False
if pca_run:
    # Load in standardized CSV
    standardized_features_df = pd.read_csv('/Users/chrisbugs/Downloads/StandardizedHockeyFeaturesV2.csv')

    # Apply PCA
    pca = PCA(n_components=10)
    pca_features = pca.fit_transform(standardized_features_df)
    pca_features_df = pd.DataFrame(pca_features, columns=[f'PC{i + 1}' for i in range(pca.n_components_)])

    print("PCA completed. Explained variance ratio:", pca.explained_variance_ratio_)
    print(pca_features_df.head(10), pca_features_df.columns)

    # plot the PCA
    pca = PCA().fit(standardized_features_df)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()

    # Save PCA result
    save_pca = False
    if save_pca:
        pca_features_df.to_csv('/Users/chrisbugs/Downloads/PCAFeaturesHockeyDataV3.csv', index=False)
        print("PCA DataFrame successfully saved to CSV.")
    