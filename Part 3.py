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
import time
import statistics


# Setting up the data frame
# GE_matrix2 = pd.read_csv("GSE34205_series_matrix_clean.txt", sep = "\t", index_col=0) # Needs the gene ID labels
meta_data = pd.read_csv("meta_data.csv")

# Gene mapping
pltform = pd.read_table("GPL570-55999.txt", comment="#", delimiter='\t') # Reading in the platform file
mapping_genes = dict(zip(pltform["ID"], pltform["Gene Symbol"])) # Creates a dictionary from the Gene ID and Gene symbols

# Filling empty values
for key,value in mapping_genes.items():
    if value is None or value != value:
        mapping_genes[key] = key # If the value is missing keep original values

# Put a check to see
features = pd.read_csv("features.csv",index_col=0)
features = features.iloc[:,:101]
# print(features)

all_transposed = features.T # Transposing the dataframe
all_transposed.index = all_transposed.index.rename("Sample_geo_accession")

# Only getting the columns needed for the merge
infections_meta = meta_data.loc[:,["Sample_geo_accession","infection_status"]]
print(infections_meta.head())

# Adding the infection status to the Gene features df
final_classifier_df = pd.merge(all_transposed,infections_meta, on="Sample_geo_accession", how = "left")
final_classifier_df.to_csv("Classifier_dataset.csv")
print(final_classifier_df.head())

# Generating the training and testing data
x = final_classifier_df.iloc[1:,1:-1] # Misses the first row (all the features) # Misses the infection status too
y = final_classifier_df["infection_status"].iloc[1:]

x_train, x_test , y_train, y_test = train_test_split(x,y,test_size = 0.3) #test data

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
    start = time.time() # For fun

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

    end = time.time()
    run_time = end - start
    print(f"Random Forrest Run: {counter + 1} Time taken: {run_time:.2f}seconds") # Just for show really
    counter += 1
# Output files

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

    # Creating a figure of the cofusion matrix
    fig = plt.figure()
    ax = fig.add_subplot(111) # Creates the plot
    cax = ax.matshow(avg_conf_matrix_df,cmap="coolwarm") # Uses the coolwarm scale (same as heatmap)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)  # these are misaligned unless you add an extra empty label
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("confused.png", format="png")

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
    best_feature_imp_df = best_feature_imp_df.sort_values(by="Average_score",ascending=False) # Sorts from most important downwards
    # Mapping gene names
    hundred_features = best_feature_imp_df.iloc[:100,:]

    best_feature_imp_df.index = best_feature_imp_df.index.map(mapping_genes)  # Maps them to the features importance dataset

    # Writing to the file
    file.write("Top 100 features (based on average importance score): \n")
    counter = 0
    for index, row in best_feature_imp_df.iloc[:100,:].iterrows():
        counter += 1
        file.write(f"{counter} GeneID: {index} , {row.values}\n")

    file.write("\n\n")

# Plotting the boxplots
num_rows, num_cols = 10, 10
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 20))
num_rows, num_cols = 10, 10
axes = axes.flatten() # So that I can plot the boxplots in an array


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

# Counters (for report)
flu_high = 0
flu_low = 0
rsv_high = 0
rsv_low = 0


for i, gene_ID in enumerate(hundred_features.index) :
    gene_data = final_classifier_df[gene_ID]
    infection_status = final_classifier_df["infection_status"] # Status
    infection_groups = gene_data.groupby(infection_status)

    # Median expression of each feature in each infection group
    flu_med = statistics.median( infection_groups.get_group("influenza"))
    rsv_med = statistics.median( infection_groups.get_group("rsv"))
    control_med = statistics.median(infection_groups.get_group("none"))

    # Assessing proportions (for report)
    if flu_med > control_med:
        flu_high +=1
    if flu_med < control_med:
        flu_low +=1
    if rsv_med > control_med:
        rsv_high += 1
    if rsv_med < control_med:
        rsv_low += 1

# For report
print(f"Influenza {flu_high} higher, {flu_low} lower")
print(f"Rsv {rsv_high} higher, {rsv_low} lower")

plt.tight_layout()

fig.savefig("Boxplots.png", format="png")

# Creating the heatmap of the 100 best features, grouped by infection status

influenza_meta = meta_data[meta_data["infection_status"] == "influenza"]
rsv_meta = meta_data[meta_data["infection_status"] == "rsv"]
control_meta = meta_data[meta_data["infection_status"] == "none"]

# Creating lists of the gene probes
influenza_gene_probes =influenza_meta["Sample_geo_accession"].tolist()
rsv_gene_probes = rsv_meta["Sample_geo_accession"].tolist()
control_gene_probes = control_meta["Sample_geo_accession"].tolist()

groups = [influenza_gene_probes,rsv_gene_probes,control_gene_probes]
files = ["influenza","rsv","control"]

# List of the 100 most important features
hundred_list = (hundred_features.index.to_list())

# Log transforming the data
logged_data = np.log(features)

# To keep the scales consistent
global_min = logged_data.min().min() # The lowest value
global_max = logged_data.max().max() # The higest value

# 3 Heatmaps sideby side
fig, axes = plt.subplots(ncols=3, figsize=(12, 8))

# Creating the heatmap
for i, group in enumerate(groups):
    temp = features.loc[:,group]
    logged_data = np.log(temp)  # Log transform
    logged_data = logged_data.loc[hundred_list] # Only gets the hundred most important features
    logged_data.index = logged_data.index.map(mapping_genes) # Y labels

    plt.figure(figsize=(12, 8))

    # Conditional formatting to make sure that the heatmaps have the right combination of scale bars and y tick labels
    if i == 2:
        heatmap_fig = sns.heatmap(logged_data, cmap="coolwarm",ax = axes[i],vmin=global_min, vmax=global_max,yticklabels=False)  # ,yticklabels=False)
        heatmap_fig.set_ylabel("")
    if i == 0:
        heatmap_fig = sns.heatmap(logged_data, cmap="coolwarm",ax = axes[i],vmin=global_min, vmax=global_max,cbar=False,yticklabels=True)  # ,yticklabels=False)
        heatmap_fig.set_yticklabels(heatmap_fig.get_yticklabels(), fontsize=5.5)
    if i == 1:
        heatmap_fig = sns.heatmap(logged_data, cmap="coolwarm",ax = axes[i],vmin=global_min, vmax=global_max,cbar=False,yticklabels=False)  # ,yticklabels=False)
        heatmap_fig.set_ylabel("")

    heatmap_fig.set_title(f"{files[i]} heatmap")


plt.tight_layout()

heatmap_fig.figure.savefig(f"heatmaps.png", format="png")

# Part 3 complete
