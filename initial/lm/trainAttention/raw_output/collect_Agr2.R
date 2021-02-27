library(tidyr)
library(dplyr)
data = read.csv("collect_Agr2.py.tsv", sep="\t")

data2 = data %>% group_by(ID, Noun, Region, Condition, predictability_weight, deletion_rate) %>% summarise(Surprisal=mean(Surprisal), ThatFraction=mean(ThatFraction)) %>% filter(Region == "V2_0")

write.table(data2, file="averages_Agr2.tsv", sep="\t")


