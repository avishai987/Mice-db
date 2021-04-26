import sqlite3
import pandas as pd
from Feature_table import Feature_table

conn = sqlite3.connect('MiceDB_test5.db')
c = conn.cursor()

command = '''CREATE TABLE IF NOT EXISTS AA_features (
            [sequence] TEXT PRIMARY KEY,[GlutamicAcid] float,
             [Glycine] float , [Serine] float , [Isoleucine] float , [Cysteine] float ,
              [Methionine] float , [NegativelyCharged] float , [Aliphatic] float , [Tyrosine] float ,
               [FrameShift] float , [Valine] float , [PositivelyCharged] float , [Asparagine] float ,
                [Lysine] float , [Polar] float , [StopCodon] float , [MolecularMass] float ,
                [NonProductive] float , [Arginine] float ,[length] integer, [Glutamine] float ,
                [Aromatic] float ,[IsoelectricPoint] float ,[Histidine] float ,[Proline] float ,
                [Tryptophan] float ,[AsparticAcid] float ,[Phenylalanine] float ,[Leucine] float ,
                [Threonine] float,[Alanine] float, [Hydrophobicity] float);

                CREATE TABLE IF NOT EXISTS MICE
             ([mice_id] INTEGER PRIMARY KEY, [experimental_group] text
              ,[survived] text);

            CREATE TABLE IF NOT EXISTS SAMPLE
             ([sample_id] INTEGER PRIMARY KEY,
             [mice_id] integer ,[time_point] integer, [source] text, FOREIGN KEY(mice_id) REFERENCES MICE(mice_id));

                CREATE TABLE DNA_AA_TRA
             ([generated_id] INTEGER PRIMARY KEY ,
             [cloneFraction] float, [nSeqImputedCDR3] text  , [aaSeqImputedCDR3] text 
             ,[sample_id] integer, FOREIGN KEY (sample_id)
             REFERENCES  SAMPLE(sample_id), FOREIGN KEY (aaSeqImputedCDR3) REFERENCES AA_features(sequence));

            CREATE TABLE IF NOT EXISTS DNA_AA_TRB
             ([generated_id] INTEGER PRIMARY KEY ,
             [cloneFraction] float, [nSeqImputedCDR3] text  , [aaSeqImputedCDR3] text,[sample_id] integer, FOREIGN KEY (sample_id)
             REFERENCES  SAMPLE(sample_id), FOREIGN KEY(aaSeqImputedCDR3)
              REFERENCES AA_features(sequence)) '''

read_DNA_AA_TRA = pd.read_csv('C:/Users/avish/raw_date/TRA/new_files/clonotypes.TRA.csv')
read_DNA_AA_TRA.to_sql('DNA_AA_TRA', conn, if_exists='append', index=False)

read_DNA_AA_TRB = pd.read_csv('C:/Users/avish/raw_date/TRB/new_files/clonotypes.TRB.csv')
read_DNA_AA_TRB.to_sql('DNA_AA_TRB', conn, if_exists='append', index=False)


read_AA_Features = pd.read_csv('out.csv')

read_AA_Features.to_sql('AA_features', conn, if_exists='append', index=False)

read_MICE = pd.read_csv('MICE.csv')
read_MICE.to_sql('MICE', conn, if_exists='append', index=False)

read_MICE = pd.read_csv('sample.csv')
read_MICE.to_sql('SAMPLE', conn, if_exists='append', index=False)


