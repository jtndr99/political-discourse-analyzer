import json
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

# 1. Load the raw data
conn = sqlite3.connect("analyses.db")
query = "SELECT id, pareto_analysis, sowell_analysis, mass_psych_analysis FROM analyses"
df = pd.read_sql_query(query, conn)

# 2. Define a helper function to safely extract numbers from the JSON text
def extract_score(json_string, key):
    try:
        # Convert the string into a Python dictionary
        data = json.loads(json_string)
        return data.get(key)
    except:
        return None

# 3. Create our 5 new metric columns by applying the helper function!
df['fox_drive_score'] = df['pareto_analysis'].apply(lambda x: extract_score(x, 'fox_drive_score'))
df['unconstrained_score'] = df['sowell_analysis'].apply(lambda x: extract_score(x, 'unconstrained_score'))
df['schmitt_intensity'] = df['sowell_analysis'].apply(lambda x: extract_score(x, 'schmitt_intensity'))
df['mimetic_tension'] = df['mass_psych_analysis'].apply(lambda x: extract_score(x, 'mimetic_tension'))
df['scapegoat_index'] = df['mass_psych_analysis'].apply(lambda x: extract_score(x, 'scapegoat_index'))

# 4. Clean up the dataframe to only keep what we need
metrics_df = df[['id', 'fox_drive_score', 'unconstrained_score', 'schmitt_intensity', 'mimetic_tension', 'scapegoat_index']].dropna()

print(f"Successfully extracted metrics for {len(metrics_df)} articles!")
print(metrics_df.head())




# 1. Grab just the 5 columns of numbers (drop the ID)
X = metrics_df[['fox_drive_score', 'unconstrained_score', 'schmitt_intensity', 'mimetic_tension', 'scapegoat_index']]

# 2. Initialize PCA to squash it down to 2 components (X and Y coordinates)
pca = PCA(n_components=2)
xy_coordinates = pca.fit_transform(X)

# 3. Add those new X and Y coordinates back into our dataframe
metrics_df['PCA_X'] = xy_coordinates[:, 0]
metrics_df['PCA_Y'] = xy_coordinates[:, 1]

# 4. Let's draw the map!
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=metrics_df, 
    x='PCA_X', 
    y='PCA_Y', 
    s=200, # Make the dots big
    color='dodgerblue'
)

# Add the Article ID next to each dot so we know who is who
for i, row in metrics_df.iterrows():
    plt.text(row['PCA_X'] + 1, row['PCA_Y'] + 1, f"ID: {row['id']}", fontsize=10)

plt.title("Political Discourse Clusters (PCA 2D Projection)")
plt.xlabel("Principal Component 1 (Biggest Variance)")
plt.ylabel("Principal Component 2 (Second Biggest Variance)")
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot as an image!
plt.savefig("data/cluster_plot.png")
print("\nPlot saved to data/cluster_plot.png! Go open it!")