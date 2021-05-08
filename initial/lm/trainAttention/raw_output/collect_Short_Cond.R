library(tidyr)
library(dplyr)
data = read.csv("collect12_NormJudg_Short_Cond.py.tsv", sep="\t")

data2 = data %>% group_by(Script, ID, Noun, Region, Condition, predictability_weight, deletion_rate) %>% summarise(Surprisal=mean(Surprisal), ThatFraction=mean(ThatFraction)) %>% filter(Region != "V3")

write.table(data2, file="averages_Short_Cond.tsv", sep="\t")


