library(tidyr)
library(dplyr)
library(ggplot2)


model_plain = read.csv("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_ZERO_EYE.py_375750655_ZeroLoss", sep="\t")

corpus = read.csv("/u/scr/mhahn/Dundee/DundeeMerged.csv", sep="\t")

matching = read.csv("matchData_EYE.py.tsv", sep="\t")

model_ids = c("EYE2.py_626041227", "EYE2.py_289477725", "EYE2.py_850742927", "EYE.py_792802677", "EYE2.py_520356935", "EYE2.py_954110712")

# 520356935 deletion_rate=0.5 predictability_weight=0.75
# 954110712 deletion_rate=0.4 predictability_weight=1.0



#EYE2.py_626041227  EYE2.py_626041227     -0.0001966976 -32.40594         Objects -0.0005218983   Subjects 1.537629e-05
#EYE2.py_289477725  EYE2.py_289477725     -0.0001373076 -22.03101         Objects -0.0006499875   Subjects -2.62582e-05
#EYE2.py_850742927  EYE2.py_850742927     -6.262999e-05 -22.14471         Objects -0.0002076424   Subjects 5.081263e-05
#EYE.py_792802677   EYE.py_792802677      -0.0002482772 -66.218           Objects -0.0004426036   Subjects 5.434549e-05
#EYE2.py_520356935  EYE2.py_520356935     -0.0001626373 -79.47759         Objects -0.0003475343   Subjects 4.902278e-05   deletion_rate=0.5 predictability_weight=0.75
#EYE2.py_954110712  EYE2.py_954110712     -6.606494e-05 -32.28462         Objects -0.0003249678   Subjects 5.411323e-05   deletion_rate=0.4 predictability_weight=1.0


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
   



   model1Obj = (lmer(FPASSD ~ SurprisalReweighted.x + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "dobj"), REML=F))
   model2Obj = (lmer(FPASSD ~ SurprisalReweighted.y + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "dobj"), REML=F))


   model1Subj = (lmer(FPASSD ~ SurprisalReweighted.x + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "nsubj"), REML=F))
   model2Subj = (lmer(FPASSD ~ SurprisalReweighted.y + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "nsubj"), REML=F))

   cat(model_id, "\t", (AIC(model1) - AIC(model2))/nrow(data), (AIC(model1) - AIC(model2)))
   cat("\t", "Objects", (AIC(model1Obj) - AIC(model2Obj))/nobs(model1Obj))
   cat("\t", "Subjects", (AIC(model1Subj) - AIC(model2Subj))/nobs(model1Subj))
   cat( "\n")



#> model1 = (lmer(FPASSD ~ SurprisalReweighted.x + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepLen < 0), REML=F))
#> model2 = (lmer(FPASSD ~ SurprisalReweighted.y + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepLen < 0), REML=F))

#> model1 = (lmer(FPASSD ~ SurprisalReweighted.x + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "dobj"), REML=F))
#> model2 = (lmer(FPASSD ~ SurprisalReweighted.y + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "dobj"), REML=F))


#> model1 = (lmer(FPASSD ~ SurprisalReweighted.x + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "nsubj"), REML=F))
#> model2 = (lmer(FPASSD ~ SurprisalReweighted.y + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "nsubj"), REML=F))


}





