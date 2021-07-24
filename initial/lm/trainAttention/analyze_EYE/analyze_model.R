library(tidyr)
library(dplyr)
library(ggplot2)



model_plain = read.csv("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_ZERO_EYE.py_375750655_ZeroLoss", sep="\t")


model = read.csv("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_EYE.py_792802677_Model", sep="\t")

model = merge(model, model_plain, by=c("Sentence", "Region", "Word"))

corpus = read.csv("/u/scr/mhahn/Dundee/DundeeMerged.csv", sep="\t")

matching = read.csv("matchData_EYE.py.tsv", sep="\t")
model = merge(model, matching, by=c("Sentence", "Region", "Word"))

data = merge(model, corpus, by=c("Itemno", "WNUM", "SentenceID", "ID" )) #, all.x=TRUE)

data$Identifier = paste(data$Sentence, data$Region)
library(lme4)

model1 = (lmer(FPASSD ~ SurprisalReweighted.x + WLEN + (1|Identifier) + (1|SUBJ), data=data))
model2 = (lmer(FPASSD ~ SurprisalReweighted.y + WLEN + (1|Identifier) + (1|SUBJ), data=data))


