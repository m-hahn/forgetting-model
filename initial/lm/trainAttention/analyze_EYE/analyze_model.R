library(tidyr)
library(dplyr)
library(ggplot2)


model_plain = read.csv("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_ZERO_EYE.py_375750655_ZeroLoss", sep="\t")

corpus = read.csv("/u/scr/mhahn/Dundee/DundeeMerged.csv", sep="\t")

matching = read.csv("matchData_EYE.py.tsv", sep="\t")

model_ids = c("EYE2.py_626041227", "EYE2.py_289477725", "EYE2.py_850742927", "EYE.py_792802677")

for(model_id in model_ids) {
model = read.csv(paste("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_", model_id, "_Model", sep=""), sep="\t") %>% group_by(Sentence, Region, Word) %>% summarise(Surprisal=mean(Surprisal), SurprisalReweighted=mean(SurprisalReweighted))

cat(model_id, " ")
model = merge(model, model_plain, by=c("Sentence", "Region", "Word"))

model = merge(model, matching, by=c("Sentence", "Region", "Word"))

data = merge(model, corpus, by=c("Itemno", "WNUM", "SentenceID", "ID" )) #, all.x=TRUE)

data$Identifier = paste(data$Sentence, data$Region)
library(lme4)

model1 = (lmer(FPASSD ~ SurprisalReweighted.x + WLEN + (1|Identifier) + (1|SUBJ), data=data, REML=F))
model2 = (lmer(FPASSD ~ SurprisalReweighted.y + WLEN + (1|Identifier) + (1|SUBJ), data=data, REML=F))

cat(model_id, "\t", (AIC(model1) - AIC(model2))/nrow(data), (AIC(model1) - AIC(model2)), "\n")
}
