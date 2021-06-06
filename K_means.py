from sklearn.cluster import KMeans
import Feature_table
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


if __name__ == '__main__':
    # Load data
    tra_df_with_Hydrophobicity = Feature_table.Feature_table("TRA_AminoAcids_aaSeqImputedCDR3_Sequence.feature_extraction.pkl").df
    tra_df = pd.read_pickle("TRA_AminoAcids_aaSeqImputedCDR3_Sequence.feature_extraction.pkl")


    ## Elbow method
    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(5, 25))
    visualizer.fit(tra_df_with_Hydrophobicity)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure

    # Load seq-smaple.id table
    DNA_AA_TRA = pd.read_csv('C:/Users/avish/raw_date/TRA/new_files/clonotypes.TRA.csv')

#######################################
    # K means - with Hydrophobicity
    kmeans = KMeans(n_clusters=15, random_state=0).fit(tra_df_with_Hydrophobicity)
    tra_df_with_Hydrophobicity['cluster'] = kmeans.labels_ #add labels to df



    # Make a new table of seq-sample_id-label
    table_with_Hydrophobicity = []
    for index, row in DNA_AA_TRA.iterrows(): #iterate all rows
        seq = row['aaSeqImputedCDR3'] # get seq
        sample_id = row['sample_id'] # get sample_id
        try:
            label = tra_df_with_Hydrophobicity.at[seq, 'cluster'] #try to search seq in the unique seqs table
            list = [seq, sample_id,label]
            table_with_Hydrophobicity.append(list)

        except:
            continue

    table_with_Hydrophobicity = pd.DataFrame(table_with_Hydrophobicity, columns = ['Sequence','sample_id','cluster'])



##############################################################
    # K means - without Hydrophobicity
    kmeans = KMeans(n_clusters=15, random_state=0).fit(tra_df)
    tra_df['cluster'] = kmeans.labels_ #add labels to df

    # Make a new table of seq-sample_id-label
    table_without_Hydrophobicity = []
    for index, row in DNA_AA_TRA.iterrows(): #iterate all rows
        seq = row['aaSeqImputedCDR3'] # get seq
        sample_id = row['sample_id'] # get sample_id
        try:
            label = tra_df.at[seq, 'cluster'] #try to search seq in the unique seqs table
            list = [seq, sample_id,label]
            table_without_Hydrophobicity.append(list)

        except:
            continue

    table_without_Hydrophobicity = pd.DataFrame(table_without_Hydrophobicity, columns = ['Sequence','sample_id','cluster'])


###############################################

#create table of sample (rows) and clusters (columns)
#with hydripobicity
    num_of_clusters = 5

#make list of 125 rows (samples) and 5 columns (clusters)
    sample_cluster_table_with_hydro = [[0] * num_of_clusters]
    for sample in range (1,124):
        sample_cluster_table_with_hydro.append([0] * num_of_clusters)

#fill list
    for index, row in table_with_Hydrophobicity.iterrows():  # iterate all rows
        sample_id = row['sample_id']
        cluster = row['cluster']
        sample_cluster_table_with_hydro[sample_id - 1] [cluster] +=1

#devide each sample (row) by it's sum
    for lst in sample_cluster_table_with_hydro:
        tot_sum = sum(lst)
        lst[:] =[x / tot_sum for x in lst]

    sample_cluster_table_with_hydro = pd.DataFrame(sample_cluster_table_with_hydro)

###############################################
# create table of sample (rows) and clusters (columns)
    #without hydripobicity
    num_of_clusters = 5

#make list of 125 rows (samples) and 5 columns (clusters)
    sample_cluster_table_without_hydro = [[0] * num_of_clusters]
    for sample in range (1,124):
        sample_cluster_table_without_hydro.append([0] * num_of_clusters)

#fill list
    for index, row in table_without_Hydrophobicity.iterrows():  # iterate all rows
        sample_id = row['sample_id']
        cluster = row['cluster']
        sample_cluster_table_without_hydro[sample_id - 1] [cluster] +=1

#devide each sample (row) by it's sum
    for lst in sample_cluster_table_without_hydro:
        tot_sum = sum(lst)
        lst[:] =[x / tot_sum for x in lst]

    sample_cluster_table_without_hydro = pd.DataFrame(sample_cluster_table_with_hydro)

