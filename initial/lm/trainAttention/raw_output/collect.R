library(tidyr)
library(dplyr)
data = read.csv("collect12_13.py.tsv", sep="\t")

data2 = data %>% group_by(Noun, Region, Condition, predictability_weight, deletion_rate) %>% summarise(Surprisal=mean(Surprisal), ThatFraction=mean(ThatFraction)) %>% filter(Region != "V3")

write.table(data2, file="averages.tsv", sep="\t")

