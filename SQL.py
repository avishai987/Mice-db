import sqlite3
import pandas as pd
from Feature_table import Feature_table

conn = sqlite3.connect('MiceDB_test2.db')
c = conn.cursor()



# Create table - AA_features
c.execute('''CREATE TABLE AA_features (
            [generated_id] INTEGER PRIMARY KEY,[GlutamicAcid] float,
             [Glycine] float , [Serine] float , [Isoleucine] float , [Cysteine] float ,
              [Methionine] float , [NegativelyCharged] float , [Aliphatic] float , [Tyrosine] float ,
               [FrameShift] float , [Valine] float , [PositivelyCharged] float , [Asparagine] float ,
                [Lysine] float , [Polar] float , [StopCodon] float , [MolecularMass] float ,
                [NonProductive] float , [Arginine] float ,[length] integer, [Glutamine] float ,
                [Aromatic] float ,[IsoelectricPoint] float ,[Histidine] float ,[Proline] float ,
                [Tryptophan] float ,[AsparticAcid] float ,[Phenylalanine] float ,[Leucine] float ,
                [Threonine] float,[Alanine] float, [Hydrophobicity] float)''')

# Create table - DNA_AA
c.execute('''CREATE TABLE DNA_AA
             ([generated_id] INTEGER PRIMARY KEY,[cloneId] integer , 
             [cloneFraction] float, [nSeqImputedCDR3] text, [aaSeqImputedCDR3] text,[sample id] integer)''')

# Create table - MICE_DNA
c.execute('''CREATE TABLE MICE_DNA
             ([generated_id] INTEGER PRIMARY KEY,[sample.id] integer ,
             [mice.id] integer,[time.point] integer, [sample.type] text, [experimental.group] text
              ,[eyeball.classification3] text)''')



read_DNA_AA = pd.read_csv ('124.csv')
read_DNA_AA.to_sql('DNA_AA', conn, if_exists='append', index = False) 

df = pd.read_pickle("TRA_AminoAcids_aaSeqImputedCDR3_Sequence.feature_extraction.pkl")

df_withHydrophoby = Feature_table("TRA_AminoAcids_aaSeqImputedCDR3_Sequence.feature_extraction.pkl")
df_withHydrophoby.df.to_sql('AA_features', conn, if_exists='append', index = False)


read_MICE_DNA = pd.read_csv ('Mice-DNA.csv')
read_MICE_DNA.to_sql('MICE_DNA', conn, if_exists='append', index = False)


