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

### Part 2
# Reading in meta data
meta_data = pd.read_csv("meta_data.csv")

# Separating the data by infection status
influenza_meta = meta_data[meta_data["infection_status"] == "influenza"]
rsv_meta = meta_data[meta_data["infection_status"] == "rsv"]
control_meta = meta_data[meta_data["infection_status"] == "none"]

# Creating lists of the gene probes
influenza_gene_probes =influenza_meta["Sample_geo_accession"].tolist()
rsv_gene_probes = rsv_meta["Sample_geo_accession"].tolist()
control_gene_probes = control_meta["Sample_geo_accession"].tolist()

# Read in the features file
features = pd.read_csv("features.csv",header=None)
# Check that it has been read in correctly

# Transpose the dataframe
features_transposed = features.iloc[:,0:101].T # Tranposes the df

# Setting the index
features_transposed.set_index(features_transposed[0], inplace = True) # Sets the gene names as the index
features_transposed = features_transposed.drop(features_transposed.columns[0],axis = 1) # Drops the orignal gene name column

# Setting the column names
features_transposed.columns = features_transposed.iloc[0]
features_transposed = features_transposed.drop(features_transposed.index[0])

# Renaming the index
features_transposed.index = features_transposed.index.rename("Sample_geo_accession") # Renames the index
features_transposed.rename(columns = {0:"Sample_geo_accession"}, inplace=True) # Renaming the gene ID table
features_transposed.to_csv("Test.csv") # For testing purposes
print(features_transposed.columns)

#  Scaling the variables

x = features_transposed
x = StandardScaler().fit_transform(x)
plt.style.use("seaborn-v0_8-darkgrid")

# Creating the PCA components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

# Creating the final PCA dataframe
final_PCA_df = pd.concat([features_transposed.reset_index(),principalDf],axis=1)

# Labels and dataframes
conditions_probes = [influenza_gene_probes,rsv_gene_probes,control_gene_probes]
conditions_labels = ["Influenza", "RSV", "Control"]
colours = ['firebrick', 'cornflowerblue', 'seagreen']
fig,ax = plt.subplots()
#
# Creating the PCA analysis for all 3 conditions
for i in range(0,3):
    temp_probe = conditions_probes[i]
    temp_condition_label = conditions_labels[i] # Sets the condition
    colour = colours[i] # Sets the colour
    one_condition_df = final_PCA_df[final_PCA_df["Sample_geo_accession"].isin(temp_probe)] # Only gets the variables that are in the list
    ax.scatter(one_condition_df.loc[:,"principal component 1"]
               , one_condition_df.loc[:,"principal component 2"],
               c = colour,s =50)
ax.set(xlabel = "Principal Component 1", ylabel = "Principal Component 2", title = "2 component PCA for infection status")
ax.legend(conditions_labels)
fig.savefig("Infection_PCA.png", format = "png")

# To look at the male vs female and ages we need to create a couple more lists
# Male and female
male_meta = meta_data[meta_data["gender"] == "M"]
female_meta = meta_data[meta_data["gender"] == "F"]
# To lists
male_gene_probes = male_meta["Sample_geo_accession"].tolist()
female_gene_probes = female_meta["Sample_geo_accession"].tolist()

# Producing the PCA for gender split
gender_probes = [male_gene_probes,female_gene_probes]
gender_labels = ["Male", "Female"]

for i in range(0, 2):
    temp_probe = gender_probes[i]
    temp_gender_label = gender_labels[i]  # Sets the gender
    colour = colours[i]  # Sets the colour
    one_gender_df = final_PCA_df[final_PCA_df["Sample_geo_accession"].isin(temp_probe)]  # Only gets the variables that are in the list
    ax.scatter(one_gender_df.loc[:, "principal component 1"],
               one_gender_df.loc[:, "principal component 2"],
               c=colour, s=50)
#
ax.set(xlabel="Principal Component 1", ylabel="Principal Component 2", title="2 component PCA for gender status")
ax.legend(gender_labels)
fig.savefig("Gender_PCA.png", format="png")


# Check if there are babies that are 6 months old
six_month_search = meta_data.query("age_months == 6.0")
print(six_month_search) # Returns some Babies

# Younger and Older (Need an equals to )
younger_meta = meta_data[meta_data["age_months"] < 6]
older_meta = meta_data[meta_data["age_months"] >= 6] #Includes babies that are 6 months old

# To lists
younger_gene_probes = younger_meta["Sample_geo_accession"].tolist()
older_gene_probes = older_meta["Sample_geo_accession"].tolist()

# Producing PCA for age split
age_probes = [younger_gene_probes, older_gene_probes]
age_labels = ["Age < 6 months", "Age ≥ 6 Months"]
#
for i in range(0, 2):
    temp_probe = age_probes[i]
    temp_age_label = age_labels[i]  # Sets the age
    colour = colours[i]  # Sets the colour
    one_age_df = final_PCA_df[final_PCA_df["Sample_geo_accession"].isin(temp_probe)]  # Only gets the variables that are in the list
    ax.scatter(one_age_df.loc[:, "principal component 1"],
               one_age_df.loc[:, "principal component 2"],
               c=colour, s=50)

ax.set(xlabel="Principal Component 1", ylabel="Principal Component 2", title="2 component PCA for age status")
ax.legend(age_labels)
fig.savefig("Age_PCA.png", format="png")

# Part 2 Complete

# Lengths for report
print(f"Young: {len(younger_gene_probes)}, Old: {len(older_gene_probes)}") # Young and "Old" (haha)
print(f"Male: {len(female_gene_probes)} Male: {len(male_gene_probes)}") # M vs F