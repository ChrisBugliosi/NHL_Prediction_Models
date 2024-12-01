import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# TODO: Need a lot more EDA visuals!!!


# Load the data
features_df = pd.read_csv('/Users/chrisbugs/Downloads/FeaturesHockeyDataV3.csv')
features_df.drop(columns='Unnamed: 0', inplace=True)

# Filter out columns that do not start with "away_" (using away columns is redundant)
filtered_columns = [col for col in features_df.columns if not col.startswith('away_')]
filtered_features_df = features_df[filtered_columns]

# Calculate the correlation matrix with filtered columns
correlation_matrix = filtered_features_df.corr()

# Set up the figure size
plt.figure(figsize=(20, 16))

# Create the heatmap using seaborn
sns.heatmap(
    correlation_matrix,
    cmap='coolwarm',  # Colormap for better contrast
    annot=False,  # Set to True if you want to display the correlation values in the cells
    fmt=".2f",  # Format for the displayed numbers
    linewidths=0.05,  # Add gridlines between cells
    cbar_kws={"shrink": 0.75},  # Customize the color bar
)

# Improve x and y-axis labels' readability
plt.xticks(rotation=90, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.subplots_adjust(top=0.95)

# Add a title
plt.title('Features Correlation Matrix', fontsize=20, pad=20)

# Display the plot
plt.show()

# Load in data
wrangled_df = pd.read_csv('/Users/chrisbugs/Downloads/WrangledHockeyDataV9.csv')

# Distribution of goalsFor
plt.figure(figsize=(12, 6))
sns.histplot(data=wrangled_df, x='goalsFor', bins=30, kde=True)
plt.title('Distribution of Goals For', fontsize=16)
plt.xlabel('Goals For', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Box plot of goalsFor
plt.figure(figsize=(16, 8))
sns.boxplot(data=wrangled_df, x='season', y='goalsFor')
plt.title('Box Plot for Outlier Detection: Goals For by Season', fontsize=16)
plt.xlabel('Season', fontsize=14)
plt.ylabel('Goals For', fontsize=14)
plt.xticks(rotation=45)
plt.show()

# Win percentage over entire dataset
win_percentage = wrangled_df.groupby('team')['home_team_win'].mean().sort_values(ascending=False)
plt.figure(figsize=(14, 7))
sns.barplot(x=win_percentage.index, y=win_percentage.values, palette='crest_r')
plt.title('Win Percentage by Team', fontsize=16)
plt.xlabel('Team', fontsize=14)
plt.ylabel('Win Percentage', fontsize=14)
plt.xticks(rotation=90)
plt.show()