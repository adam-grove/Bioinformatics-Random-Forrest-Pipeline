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
# print(meta_data.shape)
# print(meta_data.head(5))

# Splitting the data into the different groups
influenza_meta = meta_data[meta_data["infection_status"] == "influenza"]
rsv_meta = meta_data[meta_data["infection_status"] == "rsv"]
control_meta = meta_data[meta_data["infection_status"] == "none"]

# Creating lists of the gene probes
influenza_gene_probes =influenza_meta["Sample_geo_accession"].tolist()
rsv_gene_probes = rsv_meta["Sample_geo_accession"].tolist()
control_gene_probes = control_meta["Sample_geo_accession"].tolist()
# Reading in the matrix file
GE_matrix = pd.read_table("GSE34205_series_matrix_clean.txt", sep="\t", index_col=0)
original_index = GE_matrix.index # Will need to use this later
GE_matrix.reset_index(drop=True,inplace=True)



all_samples = influenza_gene_probes + rsv_gene_probes + control_gene_probes
GE_matrix = GE_matrix.reindex(columns=all_samples)

# Having a look at the matrix
print(GE_matrix.shape)
print(GE_matrix.head())
print(GE_matrix.index)

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
num_rows = len(GE_matrix)
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

# Mapping the genes
mapping_genes = dict(zip(pltform["ID"], pltform["Gene Symbol"])) # Creates a dictionary from the Gene ID and Gene symbols
Volcano_matrix = GE_matrix.copy()

Volcano_matrix["ID_REF"] = Volcano_matrix["ID_REF"].replace(mapping_genes)

print(pltform["ID"].value_counts())
print(len(mapping_genes))
# Full output File
GE_matrix.to_csv("matrix_plus_stats.csv ", index = False)
print("GE_matrix",GE_matrix)




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
rsv_significant = Volcano_matrix["Selected_flu_feature"] # Signficant plots
for i, txt in enumerate(Volcano_matrix["ID_REF"]): # Loops through each row

    # X and Y values
    x = Volcano_matrix["mean_flu_ratio"].iloc[i]
    y = -np.log(Volcano_matrix["corrected_p_value_flu"]).iloc[i]

    # If it is signficant - label the point
    if Volcano_matrix["Selected_flu_feature"].iloc[i]:  # Check for red point
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

ax.set(xlabel= "Mean log ratio", ylabel= "-log10(Corrected P-value)", title= "Rsv Volcano plot")
ax.legend(handles=[
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="firebrick", markersize=10, label="Significant"),
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="cornflowerblue", markersize=10, label="Not Significant"),
])

# Annotating significant points with labels centered above the points
rsv_significant = Volcano_matrix["Selected_rsv_feature"] # Signficant plots
for i, txt in enumerate(Volcano_matrix["ID_REF"]): # Loops through each row

    # X and Y values
    x = Volcano_matrix["mean_rsv_ratio"].iloc[i]
    y = -np.log(Volcano_matrix["corrected_p_value_rsv"]).iloc[i]

    # If it is signficant - label the point
    if Volcano_matrix["Selected_rsv_feature"].iloc[i]:  # Check for red point
        ax.annotate(txt,
                    xy=(x, y),
                    xytext=(0, 5),
                    textcoords='offset pixels',
                    ha='center', va='bottom', fontsize=5, color='black', fontweight='bold')


fig.savefig("rsv_Volcano.png", format="png")

fig.savefig("rsv_Volcano.png", format="png")


### Part 2


# Read in the features file
features = pd.read_csv("features.csv",header=None)
# Check that it has been read in correctly


# Transpose the dataframe
features_transposed = features.iloc[:,0:101].T # Tranposes# the df

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
# Check out the new data frame


# Scaling the variables


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

# Part 3 - Building a classifier model

# Setting up the data frame
GE_matrix2 = pd.read_csv("GSE34205_series_matrix_clean.txt", sep = "\t", index_col=0) # Needs the gene ID labels
all_transposed = GE_matrix2.T

all_transposed.index = all_transposed.index.rename("Sample_geo_accession")

# Only getting the columns needed for the merge
infections_meta = meta_data.loc[:,["Sample_geo_accession","infection_status"]]
print(infections_meta.head())

# Adding the infection status to the Gene features df
final_classifier_df = pd.merge(all_transposed,infections_meta, on="Sample_geo_accession", how = "left")
final_classifier_df.to_csv("Classifier_dataset.csv")
print(final_classifier_df.head())

# Generating the training and testing data
x = final_classifier_df.iloc[1:,1:-1] # Misses the first row (all the features) # Missies the infection status too
y = final_classifier_df["infection_status"].iloc[1:]

x_train, x_test , y_train, y_test = train_test_split(x,y,test_size = 0.3)

# Creating the classifier
classifer = RandomForestClassifier(n_estimators=100)

# Creating variables for random forrest and descriptive stats
all_conf_matrix = []
all_class_reports = []
all_feature_imp = []
all_f1_scores = []
counter = 0
runs = 10 # Can control how many times the random forrest runs

# Running random forrest
while counter < runs: # Runs 10 times
    # Training the classifier
    classifer.fit(x_train,y_train)
    y_pred = classifer.predict(x_test)

    # Summary Statistics
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    feature_imp = pd.Series(classifer.feature_importances_, index=x_train.columns).sort_values(ascending=False)
    f1 = f1_score(y_test,y_pred,average="weighted")


    #Appending them to the lists
    all_conf_matrix.append(conf_matrix)
    all_class_reports.append(class_report)
    feature_imp_df = feature_imp.to_frame(name='Importance')
    all_feature_imp.append(feature_imp_df)
    all_f1_scores.append(f1)


# Output files
    counter += 1
with open("Confusion_matrix.csv","w") as csv:
    for i in range(runs): # For each summary statistics
        # Saving the confusion matrix to a csv file
        labels = ["Influenza", "RSV", "Control"]
        conf_matrix_df = pd.DataFrame(all_conf_matrix[i], index=labels, columns=labels)
        conf_matrix_df.to_csv(csv , mode = "a", header=f"Run {i+1} Confusion Matrix")


with open("Classification_report_file.txt","w") as txt_file:
    for i in range(runs):
        # Writing the classification report to a file
        txt_file.write(f"Run {1+i} Classification Report\n")
        txt_file.write(all_class_reports[i])
        txt_file.write("\n\n")

with open("Importance_report_file.txt","w") as txt_file:
    for i in range(runs):
        all_feature_imp_temp = [df.copy() for df in all_feature_imp] # Stops things messing up later in the code
        temp_feature_imp = all_feature_imp_temp[i]
        # # Using more useful names for the genes of the feature importance
        temp_feature_imp.index = temp_feature_imp.index.map(mapping_genes)  # Maps them to the features importance dataset
        # Writing the importance to a file
        txt_file.write(f"Run {1+i} Importance Report\n")
        txt_file.write(str(temp_feature_imp.head(10))) # Only the top 10 most important features
        txt_file.write("\n\n")

# Summary Data
with open("Summary_report_file.txt","w") as file:
    # Average Confusion matrix score
    all_conf_matrix_arrary = np.array((all_conf_matrix)) # Turn into numpy arrary
    avg_conf_matrix_arrary = np.mean(all_conf_matrix_arrary, axis=0) # Calculate mean at each position
    labels = ["Influenza", "RSV", "Control"] # Labels

    avg_conf_matrix_df = pd.DataFrame(avg_conf_matrix_arrary, index=labels, columns=labels) # Need to turn it back into DF for labels
    file.write("Average Confusion Matrix: \n")
    file.write(avg_conf_matrix_df.to_string())
    file.write("\n\n")
    # Average f_score
    average_f1_score = sum(all_f1_scores) / runs
    average_f1_score = round(average_f1_score,4) # Round to 4 digits
    file.write("Average f1 score (calculated from weighted f1 averages): \n")
    file.write(str(average_f1_score))
    file.write("\n\n")

    # Best features
    all_feature_imp_df = pd.concat(all_feature_imp, axis=1)

    row_average = all_feature_imp_df.mean(axis=1)
    all_feature_imp_df["Average_score"] = row_average
    best_feature_imp_df = pd.DataFrame(row_average, columns=["Average_score"])
    best_feature_imp_df = best_feature_imp_df.sort_values(by="Average_score",ascending=False)
    # Mapping gene names
    hundred_features = best_feature_imp_df.iloc[:100,:]
    best_feature_imp_df.index = best_feature_imp_df.index.map(mapping_genes)  # Maps them to the features importance dataset


    file.write("Top 100 features (based on average importance score): \n")
    file.write(str(best_feature_imp_df.head(100))) #100 most important
    file.write("\n\n")

## Part 4

# Looking at the platform file
print(pltform.head())
print(pltform.columns)
print(pltform.shape)

# Creating the big box plot
print(hundred_features)
print(final_classifier_df)
# Group by the infection status

num_rows, num_cols = 10, 10
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 20))
num_rows, num_cols = 10, 10
axes = axes.flatten()



# For each index in the hundred features
for i, gene_ID in enumerate(hundred_features.index) :
    gene_data = final_classifier_df[gene_ID]
    infection_status = final_classifier_df["infection_status"] # Status
    infection_groups = gene_data.groupby(infection_status)

    # Plotting
    row_position, col_position = divmod(i, num_cols)
    data = [infection_groups.get_group("influenza"),infection_groups.get_group("rsv"),infection_groups.get_group("none")]
    axes[i].boxplot(data, labels=["influenza", "rsv", "control"])
    axes[i].set(title=mapping_genes[gene_ID])
     # Maps them to the features importance dataset



plt.tight_layout()

fig.savefig("Boxplots.png", format="png")

# Creating the heatmap
heatmap_data = GE_matrix2

#Normalising the data
scaler = StandardScaler()
normalised = scaler.fit_transform(heatmap_data)

# Creating the normalised dataframe
normalised_data = pd.DataFrame(normalised,index=GE_matrix2.index,columns=GE_matrix2.columns)

# Plotting the figure
plt.figure(figsize=(12,8))
heatmap_fig = sns.heatmap(normalised_data, cmap="coolwarm",yticklabels=False)
heatmap_fig.set_title("Normalised Gene Expression")
heatmap_fig.set(xlabel="Participant",ylabel="Gene")

# Save the figure
heatmap_fig.figure.savefig("heatmap.png", format="png")