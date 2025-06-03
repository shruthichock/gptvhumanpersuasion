import pandas as pd

# Load the data
df = pd.read_csv("annotations.csv")

# Calculate total sentences
cols = ['Greeting', 'DQ', 'NDQ', 'AV', 'LS', 'F', 'O']
df['Total Sentences'] = df[cols].sum(axis=1)

# Normalize counts by total sentences
df_pcts = df[cols].div(df['Total Sentences'], axis=0)

# Add normalized columns
df_pcts['A - Total'] = df['A - Total'] / df['Total Sentences']
df_pcts['A - Total Unique'] = df['A - Total Unique'] / df['Total Sentences']

# Add unnormalized metadata
df_pcts['Pleasantness'] = df['Pleasantness']
df_pcts['Change'] = df['Change']
df_pcts['Positive?'] = df['Change'] > 0

# Display correlations
print("Overall Correlations:\n", df.corr().round(2))
print("\nGPT Correlations:\n", df.iloc[:21].corr().round(2))
print("\nHuman Correlations:\n", df.iloc[21:].corr().round(2))

print("\Overall Correlations, Normalized:\n", df_pcts.corr().round(2))
print("\nGPT Correlations, Normalized:\n", df_pcts.iloc[:21].corr().round(2))
print("\nHuman Correlations, Normalized:\n", df_pcts.iloc[21:].corr().round(2))
