tra = read.csv("TRA.csv")
trb = read.csv("TRB.csv")

names(tra) # see columns names
names(trb) # see columns names


#boxplots feature and save the photo
boxplot <- function(selectet_fiels,table,name) {
  selevted_fields_table=  table[ ,(names(table) %in% selectet_fiels)]
  ggplot(stack(selevted_fields_table), aes(x = ind, y = values)) + geom_boxplot()+
    theme(text = element_text(size=10),
          axis.text.x = element_text(angle=90, hjust=1))
  ggsave(paste(name,".png"))

}




first_group = c('GlutamicAcid', 'Glycine', 'Serine', 'Isoleucine', 'Cysteine',
               'Methionine', 'NegativelyCharged', 'Aliphatic', 'Tyrosine',
               'FrameShift', 'Valine')

second_group = c('FrameShift', 'Valine', 
                        'PositivelyCharged', 'Asparagine', 'Lysine',
                        'Polar', 'StopCodon', 'MolecularMass', 'NonProductive', 'Arginine',
                        'length', 'Glutamine', 'Aromatic')

third_group = c('IsoelectricPoint', 'Histidine',
                'Proline', 'Tryptophan', 'AsparticAcid', 'Phenylalanine', 'Leucine',
                'Threonine', 'Alanine', 'Hydrophobicity')

#boxplot all features:

boxplot(first_group,tra,"boxplot first group tra")
boxplot(second_group,tra,"boxplot second group tra")
boxplot(third_group,tra,"boxplot third group tra")

boxplot(first_group,trb,"boxplot first group trb")
boxplot(second_group,trb,"boxplot second group trb")
boxplot(third_group,trb,"boxplot third group trb")

#variance:
sapply(tra, var)
sapply(trb, var)





all_groups = c('GlutamicAcid', 'Glycine', 'Serine', 'Isoleucine', 'Cysteine',
                'Methionine', 'NegativelyCharged', 'Aliphatic', 'Tyrosine',
             'FrameShift', 'Valine','FrameShift', 'Valine', 
                 'PositivelyCharged', 'Asparagine', 'Lysine',
                 'Polar', 'StopCodon', 'MolecularMass', 'NonProductive', 'Arginine',
                 'length', 'Glutamine', 'Aromatic','IsoelectricPoint', 'Histidine',
                'Proline', 'Tryptophan', 'AsparticAcid', 'Phenylalanine', 'Leucine',
                'Threonine', 'Alanine', 'Hydrophobicity')

#regression line for all features
for(i in all_groups){
  for(j in all_groups){
    
    ggplot(data = tra, aes_string(x =i, y = j)) +
      geom_smooth(method = 'lm')
    ggsave(path = "tra", filename = paste(i,j,".png"))
    
    }
  }

for(i in all_groups){
  for(j in all_groups){
    
    ggplot(data = trb, aes_string(x =i, y = j)) +
      geom_smooth(method = 'lm')
    ggsave(path = "trb", filename = paste(i,j,".png"))
    
  }
}
