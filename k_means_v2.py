# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:49:55 2021

@author: אבישי
"""


from sklearn.cluster import KMeans
import Feature_table
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import pandas as pd

#%% Load data


sample_mice = pd.read_csv('sample.csv')

choice = "trb without hyd"

if (choice == "tra with hyd"):
    original_feature_table = Feature_table.Feature_table("C:/Users/avish/CLionProjects/project_kmeans_data/TRA_AminoAcids_aaSeqImputedCDR3_Sequence.feature_extraction.pkl").df
    clone_fraction_table = pd.read_csv('C:/Users/avish/raw_date/TRA/new_files/clonotypes.TRA.csv')

elif (choice == "tra without hyd"):
    original_feature_table = pd.read_pickle("C:/Users/avish/CLionProjects/project_kmeans_data/TRA_AminoAcids_aaSeqImputedCDR3_Sequence.feature_extraction.pkl")
    clone_fraction_table = pd.read_csv('C:/Users/avish/raw_date/TRA/new_files/clonotypes.TRA.csv')

elif (choice == "trb with hyd"):
    original_feature_table = Feature_table.Feature_table("C:/Users/avish/CLionProjects/project_kmeans_data/TRB_AminoAcids_aaSeqImputedCDR3_Sequence.feature_extraction.pkl").df
    clone_fraction_table = pd.read_csv('C:/Users/avish/raw_date/TRB/new_files/clonotypes.TRB.csv')
    
elif (choice == "trb without hyd"):
    original_feature_table = pd.read_pickle("C:/Users/avish/CLionProjects/project_kmeans_data/TRB_AminoAcids_aaSeqImputedCDR3_Sequence.feature_extraction.pkl")
    clone_fraction_table = pd.read_csv('C:/Users/avish/raw_date/TRB/new_files/clonotypes.TRB.csv')
    
else:
    print("error")

#%% Elbow method
    # ## 
    # # Instantiate the clustering model and visualizer
    # model = KMeans()
    # visualizer = KElbowVisualizer(model, k=(5, 25))
    # visualizer.fit(original_feature_table)  # Fit the data to the visualizer
    # visualizer.show()  # Finalize and render the figure
    #
    # Load seq-smaple.id table

#%%  K means 

num_of_clusters = 10


kmeans = KMeans(n_clusters=num_of_clusters, random_state=0).fit(original_feature_table)
original_feature_table['cluster'] = kmeans.labels_ #add labels to df



# Make a new table that matches a cluster to each sequence
table = []
for index, row in clone_fraction_table.iterrows(): #iterate all rows
    seq = row['aaSeqImputedCDR3'] # get seq
    sample_id = row['sample_id'] # get sample_id
    try:
        label = original_feature_table.at[seq, 'cluster'] #try to search seq in the unique seqs table
        lst = [seq, sample_id,label]
        table.append(lst)

    except:
        continue

table = pd.DataFrame(table, columns = ['Sequence','sample_id','cluster'])

# table = pd.read_pickle("seq-sample_id-label.pkl")



#%% create table of sample (rows) and clusters (columns)



#make list of 125 rows (samples) and  columns (clusters)
sample_cluster_table = [[0] * num_of_clusters]
for sample in range (1,124):
    sample_cluster_table.append([0] * num_of_clusters)

#fill list
for index, row in table.iterrows():  # iterate all rows
    sample_id = row['sample_id']
    cluster = row['cluster']
    sample_cluster_table[sample_id - 1] [cluster] +=1

#divide each sample (row) by it's sum
for lst in sample_cluster_table:
    tot_sum = sum(lst)
    lst[:] =[x / tot_sum for x in lst]

sample_cluster_table = pd.DataFrame(sample_cluster_table)
# sample_cluster_table = pd.read_pickle("./sample_clusters_hydro.pkl")
#sample_cluster_table.to_csv("./pickles/***.csv")



#%% coorelation matrix 
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
tra_corr = pd.read_csv('C:/Users/avish/CLionProjects/project_kmeans_data/tra_corr.csv')
df = tra_corr.iloc[1:13, 1:13]
corrMatrix = df.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()
plt.savefig('C:/Users/avish/CLionProjects/project_kmeans_data/foo.png')

