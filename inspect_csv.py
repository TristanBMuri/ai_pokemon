import pandas as pd

# Load the CSV without header to see the raw grid
df = pd.read_csv('data/Default Mode Bosses v4.1 (with EVs) - Radical Red - Kanto Leaders.csv', header=None)

# Print the first 20 rows and first 20 columns to understand the grid
print(df.iloc[:20, :20].to_string())

# Check where "Geodude-A" is
print("\nLocation of 'Geodude-A':")
print(df[df.eq("Geodude-A").any(axis=1)])
