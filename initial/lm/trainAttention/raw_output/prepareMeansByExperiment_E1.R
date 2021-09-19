library(dplyr)
library(tidyr)
data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_E1.py.tsv", sep="\t")

write.table(data %>% group_by(Noun, Region, Condition, Script, ID, predictability_weight, deletion_rate, autoencoder, lm) %>% summarise(Surprisal=mean(Surprisal), SurprisalReweighted=mean(SurprisalReweighted), ThatFraction=mean(ThatFraction), ThatFractionReweighted=mean(ThatFractionReweighted)), file="prepareMeansByExperiment_E1.R.tsv", sep="\t")
write.table(data %>% group_by(Noun, Region, Condition, Script, ID, predictability_weight, deletion_rate, autoencoder, lm) %>% summarise(Surprisal=mean(Surprisal), SurprisalReweighted=mean(SurprisalReweighted), ThatFraction=mean(ThatFraction), ThatFractionReweighted=mean(ThatFractionReweighted)), file="prepareMeansByExperiment_E1_ByStimuli.R.tsv", sep="\t")


