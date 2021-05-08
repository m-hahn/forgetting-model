library(tidyr)
library(dplyr)
data = read.csv("collect12_NormJudg_Short_Cond_W_GPT2.py.tsv", sep="\t")

data2 = data %>% group_by(Script, ID, Noun, Region, Condition, predictability_weight, deletion_rate) %>% summarise(Surprisal=mean(Surprisal), ThatFraction=mean(ThatFraction), SurprisalReweighted=mean(SurprisalReweighted), ThatFractionReweighted=mean(ThatFractionReweighted)) %>% filter(Region != "V3")

write.table(data2, file="averages_Short_Cond_W_GPT2.tsv", sep="\t")


