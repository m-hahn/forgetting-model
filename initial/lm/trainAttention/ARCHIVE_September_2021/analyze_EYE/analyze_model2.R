library(tidyr)
library(dplyr)
library(ggplot2)


model_plain = read.csv("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_ZERO_EYE-T.py_495510756_ZeroLoss", sep="\t")

corpus = read.csv("/u/scr/mhahn/Dundee/DundeeMerged.csv", sep="\t")



matching = read.csv("matchData_EYE.py.tsv", sep="\t")
wordfreq = read.csv("dundee-bnc-frequencies.tsv", sep="\t")

matching = merge(matching, wordfreq %>% rename(Word = LowerCaseToken, BNCFrequency=Frequency), all.x=TRUE)

tokenized = read.csv("/u/scr/mhahn/Dundee/DundeeTreebankTokenized2.csv", sep="\t")

model_ids = c("EYE2.py_626041227", "EYE2.py_289477725", "EYE2.py_850742927", "EYE.py_792802677", "EYE2.py_520356935", "EYE2.py_954110712")

#
#for(model_id in model_ids) {
model_id = 477893624
model_id = 551995690
model_id=200185521
model_id=471958125
#4.6M    /u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_EYE2-T.py_210961158_Model
#4.6M    /u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_EYE2-T.py_838997785_Model
#4.6M    /u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_EYE2-T.py_899225807_Model
model_id=838997785

model_id=899225807
models = read.csv("models.tsv", sep=" ")
for(i in (1:nrow(models))) {
   model_id = models$ID[[i]]
   model = read.csv(paste("/u/scr/mhahn/reinforce-logs-both-short/calibration-full-logs-tsv/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN3Stims_3_W_GPT2M_EYE2-T.py_", model_id, "_Model", sep=""), sep="\t")
 if(!("TokenLower" %in% names(model))) {
    cat("ERROR", model_id, "\n")
 } else {
   model = model %>% group_by(Sentence, Region, TokenLower, Itemno, WNUM, SentenceID, ID, WORD, tokenInWord) %>% summarise(Surprisal=mean(Surprisal), SurprisalReweighted=mean(SurprisalReweighted))
   cat(model_id, " ")
   model = merge(model, model_plain, by=c("Sentence", "Region", "TokenLower", "Itemno", "WNUM", "SentenceID", "ID", "WORD", "tokenInWord"), all=TRUE)
   model = merge(model, matching, by=c("Itemno", "WNUM", "SentenceID", "ID", "WORD"), all=TRUE)
#model[is.na(model$Surprisal.x),]    # words with punctuation have no surprisal
   data = merge(model, corpus, by=c("Itemno", "WNUM", "SentenceID", "ID" ), all=TRUE) #, all.x=TRUE)
# head(data[is.na(data$Surprisal.x),])  Tokens without prediction. Note WORD.x vs WORD.y
   data$Identifier = paste(data$Sentence.y, data$Region.y)
   library(lme4)
data$following_WNUM = data$WNUM+1
data$FIXATED = (data$FPASSD > 0)


#nrow( data %>% filter(is.na(Surprisal.x), !is.na(Surprisal.y)))
#nrow( data %>% filter(!is.na(Surprisal.x), is.na(Surprisal.y)) )      
  
data = data %>% filter(!is.na(SurprisalReweighted.x), !is.na(SurprisalReweighted.y))

#> nrow(unique(data %>% select(Itemno, WNUM, SUBJ)))
#[1] 229143
#> nrow((data %>% select(Itemno, WNUM, SUBJ)))
#


#> nrow(data)
#[1] 229148

data[duplicated((data %>% select(Itemno, WNUM, SUBJ))),]
# for reason not clear to me right now, observations for a word [our] are duplicated for five different participants


data = data[!duplicated((data %>% select(Itemno, WNUM, SUBJ))),]


data$LogBNCFreq = log(data$BNCFrequency)

data_previous = data %>% select(following_WNUM, LogBNCFreq, SUBJ, Itemno, WLEN, SurprisalReweighted.x, SurprisalReweighted.y, FIXATED) %>% rename(WNUM=following_WNUM, previous_WLEN=WLEN, previous_SurprisalReweighted.x=SurprisalReweighted.x,  previous_SurprisalReweighted.y=SurprisalReweighted.y, previous_FIXATED=FIXATED, previous_LogBNCFreq=LogBNCFreq)


data = merge(data, data_previous, by=c("Itemno", "WNUM", "SUBJ"), all.x=TRUE)
#> nrow(data) # TODO something is wrong
#[1] 359250

data = data %>% filter(FPASSD > 0)

data$SurprisalReweighted.x.Resid = resid(lm(SurprisalReweighted.x ~ LogBNCFreq + WLEN, data=data, na.action=na.exclude))
data$SurprisalReweighted.y.Resid = resid(lm(SurprisalReweighted.y ~ LogBNCFreq + WLEN, data=data, na.action=na.exclude))
data$SurprisalReweighted.x.Resid = resid(lm(previous_SurprisalReweighted.x ~ previous_LogBNCFreq + previous_WLEN, data=data, na.action=na.exclude))
data$SurprisalReweighted.y.Resid = resid(lm(previous_SurprisalReweighted.y ~ previous_LogBNCFreq + previous_WLEN, data=data, na.action=na.exclude))


data$SurprisalReweighted.x = data$SurprisalReweighted.x - mean(data$SurprisalReweighted.x, na.rm=TRUE)
data$SurprisalReweighted.y = data$SurprisalReweighted.y - mean(data$SurprisalReweighted.y, na.rm=TRUE)
data$WLEN = data$WLEN - mean(data$WLEN, na.rm=TRUE)
data$LogBNCFreq = data$LogBNCFreq - mean(data$LogBNCFreq, na.rm=TRUE)

model1 = (lmer(FPASSD ~ SurprisalReweighted.x + WLEN + LogBNCFreq + WLEN + LogBNCFreq + (1|Identifier) + (1|SUBJ), data=data, REML=F))
model2 = (lmer(FPASSD ~ SurprisalReweighted.y + WLEN + LogBNCFreq + WLEN + LogBNCFreq + (1|Identifier) + (1|SUBJ), data=data, REML=F))
#model1 = (lmer(FPASSD ~ SurprisalReweighted.x + WLEN + LogBNCFreq + WLEN*LogBNCFreq + (1|Identifier) + (1|SUBJ), data=data, REML=F))
#model2 = (lmer(FPASSD ~ SurprisalReweighted.y + WLEN + LogBNCFreq + WLEN*LogBNCFreq + (1|Identifier) + (1|SUBJ), data=data, REML=F))

   cat(model_id, "\t", (AIC(model1) - AIC(model2))/nrow(data), (AIC(model1) - AIC(model2)), models$deletion_rate[[i]], models$predictability_weight[[i]])
   sink("analyze_model2.R.tsv", append=TRUE)
   cat(model_id, (AIC(model1) - AIC(model2))/nrow(data), (AIC(model1) - AIC(model2)), models$deletion_rate[[i]], models$predictability_weight[[i]], sep="\t")
   sink()
   cat( "\n")

}
}


crash()

   model1 = (lmer(FPASSD ~ SurprisalReweighted.x + WLEN + previous_WLEN + previous_SurprisalReweighted.x + previous_FIXATED + LogBNCFreq + previous_LogBNCFreq + (1|Identifier) + (1|SUBJ), data=data, REML=F))
   model2 = (lmer(FPASSD ~ SurprisalReweighted.y + WLEN + previous_WLEN + previous_SurprisalReweighted.y + previous_FIXATED + LogBNCFreq + previous_LogBNCFreq + (1|Identifier) + (1|SUBJ), data=data, REML=F))
   


   anova(model2, model1)

   #model1Obj = (lmer(FPASSD ~ SurprisalReweighted.x + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "dobj"), REML=F))
  # model2Obj = (lmer(FPASSD ~ SurprisalReweighted.y + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "dobj"), REML=F))


 #  model1Subj = (lmer(FPASSD ~ SurprisalReweighted.x + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "nsubj"), REML=F))
#   model2Subj = (lmer(FPASSD ~ SurprisalReweighted.y + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "nsubj"), REML=F))

   cat(model_id, "\t", (AIC(model1) - AIC(model2))/nrow(data), (AIC(model1) - AIC(model2)))
#   cat("\t", "Objects", (AIC(model1Obj) - AIC(model2Obj))/nobs(model1Obj))
 #  cat("\t", "Subjects", (AIC(model1Subj) - AIC(model2Subj))/nobs(model1Subj))
   cat( "\n")

# use Python script to
# (1) assemble the list of models
# (2) assemble word frequencies (BNC and Wikipedia and others?)
# (3) assemble DLT predictions


#> model1 = (lmer(FPASSD ~ SurprisalReweighted.x + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepLen < 0), REML=F))
#> model2 = (lmer(FPASSD ~ SurprisalReweighted.y + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepLen < 0), REML=F))

#> model1 = (lmer(FPASSD ~ SurprisalReweighted.x + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "dobj"), REML=F))
#> model2 = (lmer(FPASSD ~ SurprisalReweighted.y + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "dobj"), REML=F))


#> model1 = (lmer(FPASSD ~ SurprisalReweighted.x + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "nsubj"), REML=F))
#> model2 = (lmer(FPASSD ~ SurprisalReweighted.y + WLEN + (1|Identifier) + (1|SUBJ), data=data %>% filter(DepRel == "nsubj"), REML=F))


#}





