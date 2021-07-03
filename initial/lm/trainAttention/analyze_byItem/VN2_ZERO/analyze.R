gulo = read.csv("/u/scr/mhahn/log-gulordava.tsv", sep="\t")

txl = read.csv("/u/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN2Stims_3_W_TXL_ZERO.py_858805767_ZeroLoss", sep="\t")

gpt2m = read.csv("/sailhome/mhahn/scr/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN2Stims_3_W_GPT2M_ZERO.py_503159126_ZeroLoss", sep="\t")

gpt2xl = read.csv("/sailhome/mhahn/scr/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN2Stims_3_W_GPT2XL_ZERO.py_931606655_ZeroLoss", sep="\t")


library(ggplot2)
library(tidyr)
library(dplyr)
library(lme4)


model = lmer(SurprisalReweighted ~ RC + compatible + (1|Noun) + (1+compatible|Item), data=gulo %>% filter(!grepl("NoSC", Condition)) %>% mutate(RC= grepl("RC", Condition), compatible = grepl("_co", Condition)))
write.table(coef((model))$Item, file="output/compatibility-slots-gulordava.tsv")

slopes = coef((model))$Item
slopes$Item = rownames(slopes)
slopes$Gulordava = slopes$compatibleTRUE

slopes_ = slopes

model = lmer(SurprisalReweighted ~ RC + compatible + (1|Noun) + (1+compatible|Item), data=txl %>% filter(!grepl("NoSC", Condition)) %>% mutate(RC= grepl("RC", Condition), compatible = grepl("_co", Condition)))
write.table(coef((model))$Item, file="output/compatibility-slots-txl.tsv")

slopes = coef((model))$Item
slopes$Item = rownames(slopes)
slopes$TXL = slopes$compatibleTRUE

slopes_ = merge(slopes, slopes_, by=c("Item"))

model = lmer(SurprisalReweighted ~ RC + compatible + (1|Noun) + (1+compatible|Item), data=gpt2m %>% filter(!grepl("NoSC", Condition)) %>% mutate(RC= grepl("RC", Condition), compatible = grepl("_co", Condition)))
write.table(coef((model))$Item, file="output/compatibility-slots-gpt2m.tsv")

slopes = coef((model))$Item
slopes$Item = rownames(slopes)
slopes$GPT2M = slopes$compatibleTRUE

slopes_ = merge(slopes, slopes_, by=c("Item"))


model = lmer(SurprisalReweighted ~ RC + compatible + (1|Noun) + (1+compatible|Item), data=gpt2xl %>% filter(!grepl("NoSC", Condition)) %>% mutate(RC= grepl("RC", Condition), compatible = grepl("_co", Condition)))
write.table(coef((model))$Item, file="output/compatibility-slots-gpt2xl.tsv")

slopes = coef((model))$Item
slopes$Item = rownames(slopes)
slopes$GPT2XL = slopes$compatibleTRUE

slopes_ = merge(slopes, slopes_, by=c("Item"))





