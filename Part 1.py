# Imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
from sklearn.metrics import f1_score
import seaborn as sns

### Part 1

# Start by reading in the metadata and the platform file
meta_data = pd.read_csv("meta_data.csv")
pltform = pd.read_table("GPL570-55999.txt", comment="#", delimiter='\t') # Need to ignore the columns

# Having a look at the metadata
print(meta_data.shape)
print(meta_data.head(5))

# Splitting the data into the different groups
influenza_meta = meta_data[meta_data["infection_status"] == "influenza"]
rsv_meta = meta_data[meta_data["infection_status"] == "rsv"]
control_meta = meta_data[meta_data["infection_status"] == "none"]

# Creating lists of the gene probes
influenza_gene_probes =influenza_meta["Sample_geo_accession"].tolist()
rsv_gene_probes = rsv_meta["Sample_geo_accession"].tolist()
control_gene_probes = control_meta["Sample_geo_accession"].tolist()

# Reading in the matrix file - I've had some problems with the index column and this is the only way it works
GE_matrix = pd.read_table("GSE34205_series_matrix_clean.txt", sep="\t", index_col=0)
original_index = GE_matrix.index # Will need to use this later
GE_matrix.reset_index(drop=True,inplace=True) # Resets the index

# Reindexing the samples
all_samples = influenza_gene_probes + rsv_gene_probes + control_gene_probes # Creates the list in order
GE_matrix = GE_matrix.reindex(columns=all_samples) # Reshuffles the columns to be in order

# Having a look at the matrix
print(GE_matrix.shape)
print(GE_matrix.head())


# Assigning new values - Copied from the assignment sheet with my own value names
GE_matrix = GE_matrix.assign(mean_rsv_ratio=np.zeros(len(GE_matrix)))
GE_matrix = GE_matrix.assign(mean_flu_ratio=np.zeros(len(GE_matrix)))
GE_matrix = GE_matrix.assign(p_value_rsv=np.zeros(len(GE_matrix)))
GE_matrix = GE_matrix.assign(corrected_p_value_rsv=np.zeros(len(GE_matrix)))
GE_matrix = GE_matrix.assign(p_value_flu=np.zeros(len(GE_matrix)))
GE_matrix = GE_matrix.assign(corrected_p_value_flu=np.zeros(len(GE_matrix)))
GE_matrix["Selected_flu_feature"] = False
GE_matrix["Selected_rsv_feature"] = False
GE_matrix["Selected_feature"] = False

# Subsetting data
influenza_matrix = GE_matrix[influenza_gene_probes] # Flu
rsv_matrix = GE_matrix[rsv_gene_probes] # Rsv
print(rsv_matrix.shape)


# Checking subsets have been done correctly
# print(influenza_matrix.head())
# print(len(influenza_gene_probes))
# print(rsv_matrix.head())
# print(len(rsv_gene_probes))


## Calculating P values and corrected P values
# Influenza patients Mean and P value
flu_mean_values = np.log(influenza_matrix).mean(axis=1) # P value
t_stat, flu_p_values = stats.ttest_1samp(np.log(influenza_matrix),0.0,axis=1) # T test
GE_matrix["mean_flu_ratio"] = flu_mean_values
GE_matrix["p_value_flu"] = flu_p_values

# Bonferroni correction for influenza patients (p value * number of tests)
num_rows = len(GE_matrix)
GE_matrix["corrected_p_value_flu"] = GE_matrix["p_value_flu"] * num_rows

# RSV mean and P value
rsv_mean_values = np.log(rsv_matrix).mean(axis=1) # P value
t_stat, rsv_p_values = stats.ttest_1samp(np.log(rsv_matrix),0.0,axis=1) # T test
GE_matrix["mean_rsv_ratio"] = rsv_mean_values
GE_matrix["p_value_rsv"] = rsv_p_values

# Bonferroni correction for RSV patients
GE_matrix["corrected_p_value_rsv"] = GE_matrix["p_value_rsv"] * num_rows

# Updating the feature columns - Need to make sure that it works for -1 and +1
for i in range(GE_matrix.shape[0]):
    if abs(GE_matrix["mean_flu_ratio"][i]) > 1 and GE_matrix["corrected_p_value_flu"][i] < 0.05: # Significant flu features
        GE_matrix.loc[i,"Selected_flu_feature"] = True
        GE_matrix.loc[i, "Selected_feature"] = True
    if abs(GE_matrix["mean_rsv_ratio"][i]) > 1 and GE_matrix["corrected_p_value_rsv"][i] < 0.05: # Significant rsv features
        GE_matrix.loc[i,"Selected_rsv_feature"] = True
        GE_matrix.loc[i, "Selected_feature"] = True

# having a look to see if this has updated correctly
print(GE_matrix["Selected_flu_feature"].value_counts())
print(GE_matrix["Selected_rsv_feature"].value_counts())


# Putting the index back in - Fixing the index problem I have had earlier
GE_matrix.set_index(original_index, inplace=True)
GE_matrix.insert(0,"ID_REF",original_index)

# Full output File
GE_matrix.to_csv("matrix_plus_stats.csv ", index = False) # Does not have the index column
#print("GE_matrix",GE_matrix)


# Mapping the genes
mapping_genes = dict(zip(pltform["ID"], pltform["Gene Symbol"])) # Creates a dictionary from the Gene ID and Gene symbols

# Filling empty values
for key,value in mapping_genes.items():
    if value is None or value != value:
        mapping_genes[key] = key # Fills empty values with the original gene name

# print(Volcano_matrix.index)

# Features File
Features_matrix = GE_matrix[GE_matrix["Selected_feature"] == True] # Subsetting
Features_matrix.to_csv("features.csv", index = False) # Saving

Volcano_matrix = GE_matrix.copy() # So that it does not affect the original matrix

Volcano_matrix["ID_REF"] = Volcano_matrix["ID_REF"].replace(mapping_genes) # Replaces the gene names with gene symbols


# Volcano Plot for flu
print("Plotting Flu Volcano")
plt.style.use("seaborn-v0_8-darkgrid") # Going to use this for consistency
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(Volcano_matrix["mean_flu_ratio"], -np.log(Volcano_matrix["corrected_p_value_flu"]),
           c = np.where(Volcano_matrix["Selected_flu_feature"], "firebrick","cornflowerblue")) # Red and blue so that it stands out
ax1.set(xlabel= "Mean log ratio", ylabel= "-log10(Corrected P-value)", title= "Influenza Volcano plot")
ax1.legend(handles=[
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="firebrick", markersize=10, label="Significant"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="cornflowerblue", markersize=10, label="Not Significant"),
]) # Legend

# So that the axis are the same for both
ax1.set_ylim(-15, 55)
ax1.set_xlim(-3, 5)

# Annotating the significant points
for i, txt in enumerate(Volcano_matrix["ID_REF"]): # Loops through each row (gets the Gene identifier as text)
    # X and Y values
    x = Volcano_matrix["mean_flu_ratio"].iloc[i]
    y = -np.log(Volcano_matrix["corrected_p_value_flu"]).iloc[i]

    # If it is significant - label the point
    if abs(x) > 2 and y > 1.30:  # Check for significance (only the most significant)
        ax1.annotate(txt,
                    xy=(x, y),
                    xytext=(0, 5),
                    textcoords='offset pixels',
                    ha='center', va='bottom', fontsize=5, color='black', fontweight='bold') # Label formatting



# Volcano Plot for rsv
print("Plotting RSV Volcano")
ax2.scatter(Volcano_matrix["mean_rsv_ratio"], -np.log(Volcano_matrix["corrected_p_value_rsv"]),
           c = np.where(Volcano_matrix["Selected_rsv_feature"], "firebrick","cornflowerblue"))

ax2.set( xlabel= "Mean RSV expression ", ylabel= "-log10(Corrected P-value)", title= "Rsv Volcano plot")
ax2.legend(handles=[
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="firebrick", markersize=10, label="Significant"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="cornflowerblue", markersize=10, label="Not Significant"),
])

# So that the axis are the same
ax2.set_ylim(-15, 55)
ax2.set_xlim(-3, 5)

# Annotating the most  significant points with labels centered above the points
for i, txt in enumerate(Volcano_matrix["ID_REF"]): # Loops through each row

    # X and Y values
    x = Volcano_matrix["mean_rsv_ratio"].iloc[i]
    y = -np.log(Volcano_matrix["corrected_p_value_rsv"]).iloc[i]

    # If it is signficant - label the point
    if abs(x)>2 and y > 1.30 :  # Check for red point
        ax2.annotate(txt,
                    xy=(x, y),
                    xytext=(0, 5),
                    textcoords='offset pixels',
                    ha='center', va='bottom', fontsize=5, color='black', fontweight='bold')


fig.savefig("Volcanos.png", format="png") # Volcanos side by side

# Part 1 complete

