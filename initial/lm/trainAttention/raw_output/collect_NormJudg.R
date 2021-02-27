library(tidyr)
library(dplyr)
data = read.csv("collect12_NormJudg.py.tsv", sep="\t")

data2 = data %>% group_by(ID, Noun, Region, Condition, predictability_weight, deletion_rate) %>% summarise(Surprisal=mean(Surprisal), ThatFraction=mean(ThatFraction)) %>% filter(Region != "V3")

write.table(data2, file="averages_NormJudg.tsv", sep="\t")


