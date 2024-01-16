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
#
#
# Having a look at the metadata
print(meta_data.shape)
print(meta_data.head(5))
#
# Splitting the data into the different groups
influenza_meta = meta_data[meta_data["infection_status"] == "influenza"]
rsv_meta = meta_data[meta_data["infection_status"] == "rsv"]
control_meta = meta_data[meta_data["infection_status"] == "none"]

# Creating lists of the gene probes
influenza_gene_probes =influenza_meta["Sample_geo_accession"].tolist()
rsv_gene_probes = rsv_meta["Sample_geo_accession"].tolist()
control_gene_probes = control_meta["Sample_geo_accession"].tolist()
#
# Reading in the matrix file
GE_matrix = pd.read_table("GSE34205_series_matrix_clean.txt", sep="\t", index_col=0)
original_index = GE_matrix.index # Will need to use this later
GE_matrix.reset_index(drop=True,inplace=True)

# Reindexing the samples
all_samples = influenza_gene_probes + rsv_gene_probes + control_gene_probes
GE_matrix = GE_matrix.reindex(columns=all_samples)

# Having a look at the matrix
print(GE_matrix.shape)
print(GE_matrix.head())


# Assigning new values
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
influenza_matrix = GE_matrix[influenza_gene_probes]
rsv_matrix = GE_matrix[rsv_gene_probes]
print(rsv_matrix.shape)


# Checking subsets have been done correctly
# print(influenza_matrix.head())
# print(len(influenza_gene_probes))
# print(rsv_matrix.head())
# print(len(rsv_gene_probes))


## Calculating P values and corrected P values
# Influenza patients Mean and P value
flu_mean_values = np.log(influenza_matrix).mean(axis=1)
t_stat, flu_p_values = stats.ttest_1samp(np.log(influenza_matrix),0.0,axis=1)
GE_matrix["mean_flu_ratio"] = flu_mean_values
GE_matrix["p_value_flu"] = flu_p_values

# Bonferroni correction for influenza patients
num_rows = len(GE_matrix)
GE_matrix["corrected_p_value_flu"] = GE_matrix["p_value_flu"] * num_rows

# RSV mean and P value
rsv_mean_values = np.log(rsv_matrix).mean(axis=1)
t_stat, rsv_p_values = stats.ttest_1samp(np.log(rsv_matrix),0.0,axis=1)
GE_matrix["mean_rsv_ratio"] = rsv_mean_values
GE_matrix["p_value_rsv"] = rsv_p_values

# Bonferroni correction for RSV patients
GE_matrix["corrected_p_value_rsv"] = GE_matrix["p_value_rsv"] * num_rows

# Updating the feature columns - Need to make sure that it works for -1 and +1
for i in range(GE_matrix.shape[0]):
    if abs(GE_matrix["mean_flu_ratio"][i]) > 1 and GE_matrix["corrected_p_value_flu"][i] < 0.05:
        GE_matrix.loc[i,"Selected_flu_feature"] = True
        GE_matrix.loc[i, "Selected_feature"] = True
    if abs(GE_matrix["mean_rsv_ratio"][i]) > 1 and GE_matrix["corrected_p_value_rsv"][i] < 0.05:
        GE_matrix.loc[i,"Selected_rsv_feature"] = True
        GE_matrix.loc[i, "Selected_feature"] = True

# having a look to see if this has updated correctly
print(GE_matrix["Selected_flu_feature"].value_counts())
print(GE_matrix["Selected_rsv_feature"].value_counts())


# Putting the index back in
GE_matrix.set_index(original_index, inplace=True)
GE_matrix.insert(0,"ID_REF",original_index)

# Full output File
GE_matrix.to_csv("matrix_plus_stats.csv ", index = False)
print("GE_matrix",GE_matrix)


# Mapping the genes
mapping_genes = dict(zip(pltform["ID"], pltform["Gene Symbol"])) # Creates a dictionary from the Gene ID and Gene symbols

# Filling empty values
for key,value in mapping_genes.items():
    if value is None or value != value:
        print("Empty found for key:", key)
        mapping_genes[key] = key

Volcano_matrix = GE_matrix.copy()

Volcano_matrix["ID_REF"] = Volcano_matrix["ID_REF"].replace(mapping_genes)
print(Volcano_matrix.index)
#
#
#
#
#
# Features File
Features_matrix = GE_matrix[GE_matrix["Selected_feature"] == True]
Features_matrix.to_csv("features.csv", index = False)


# Volcano Plot for flu
plt.style.use("seaborn-v0_8-darkgrid")
fig,ax  = plt.subplots()
ax.scatter(Volcano_matrix["mean_flu_ratio"], -np.log(Volcano_matrix["corrected_p_value_flu"]),
           c = np.where(Volcano_matrix["Selected_flu_feature"], "firebrick","cornflowerblue"))
ax.set(xlabel= "Mean log ratio", ylabel= "-log10(Corrected P-value)", title= "Influenza Volcano plot")
ax.legend(handles=[
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="firebrick", markersize=10, label="Significant"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="cornflowerblue", markersize=10, label="Not Significant"),
])

# Annotating the significant points
for i, txt in enumerate(Volcano_matrix["ID_REF"]): # Loops through each row (gets the Gene identifier as text)
    # X and Y values
    x = Volcano_matrix["mean_flu_ratio"].iloc[i]
    y = -np.log(Volcano_matrix["corrected_p_value_flu"]).iloc[i]

    # If it is significant - label the point
    if abs(x) > 2 and y > 1.30:  # Check for significance
        ax.annotate(txt,
                    xy=(x, y),
                    xytext=(0, 5),
                    textcoords='offset pixels',
                    ha='center', va='bottom', fontsize=5, color='black', fontweight='bold')


fig.savefig("flu_Volcano.png", format="png")

# Volcano Plot for rsv
fig,ax  = plt.subplots()
ax.scatter(Volcano_matrix["mean_rsv_ratio"], -np.log(Volcano_matrix["corrected_p_value_rsv"]),
           c = np.where(Volcano_matrix["Selected_rsv_feature"], "firebrick","cornflowerblue"))

ax.set(xlabel= "Mean RSV expression ", ylabel= "-log10(Corrected P-value)", title= "Rsv Volcano plot")
ax.legend(handles=[
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="firebrick", markersize=10, label="Significant"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="cornflowerblue", markersize=10, label="Not Significant"),
])

# Annotating the most  significant points with labels centered above the points
for i, txt in enumerate(Volcano_matrix["ID_REF"]): # Loops through each row

    # X and Y values
    x = Volcano_matrix["mean_rsv_ratio"].iloc[i]
    y = -np.log(Volcano_matrix["corrected_p_value_rsv"]).iloc[i]

    # If it is signficant - label the point
    if abs(x)>2 and y > 1.30 :  # Check for red point
        ax.annotate(txt,
                    xy=(x, y),
                    xytext=(0, 5),
                    textcoords='offset pixels',
                    ha='center', va='bottom', fontsize=5, color='black', fontweight='bold')


fig.savefig("rsv_Volcano.png", format="png")
#
#
