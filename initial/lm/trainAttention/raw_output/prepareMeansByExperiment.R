library(dplyr)
library(tidyr)
data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN3.py.tsv", sep="\t")

data$Experiment = ifelse(grepl("245_", data$Item), "Experiment1", "Experiment2")

write.table(data %>% group_by(Noun, Experiment, Region, Condition, Script, ID, predictability_weight, deletion_rate, autoencoder, lm) %>% summarise(Surprisal=mean(Surprisal), SurprisalReweighted=mean(SurprisalReweighted), ThatFraction=mean(ThatFraction), ThatFractionReweighted=mean(ThatFractionReweighted)), file="prepareMeansByExperiment.R.tsv", sep="\t")


