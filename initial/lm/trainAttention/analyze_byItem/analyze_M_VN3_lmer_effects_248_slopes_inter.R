library(tidyr)
library(dplyr)
library(lme4)


data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_VN3.py.tsv", sep="\t")  %>% filter(Region == "V1_0")



nounFreqs = read.csv("../../../../../forgetting/corpus_counts/wikipedia/results/results_counts4.py.tsv", sep="\t")
nounFreqs$LCount = log(1+nounFreqs$Count)
nounFreqs$Condition = paste(nounFreqs$HasThat, nounFreqs$Capital, "False", sep="_")
nounFreqs = as.data.frame(unique(nounFreqs) %>% select(Noun, Condition, LCount) %>% group_by(Noun) %>% spread(Condition, LCount)) %>% rename(noun = Noun)


nounFreqs2 = read.csv("../../../../../forgetting/corpus_counts/wikipedia/results/archive/perNounCounts.csv") %>% mutate(X=NULL, ForgettingVerbLogOdds=NULL, ForgettingMiddleVerbLogOdds=NULL) %>% rename(noun = Noun) %>% mutate(True_True_True=NULL,True_False_True=NULL)

nounFreqs = unique(rbind(nounFreqs, nounFreqs2))
nounFreqs = nounFreqs[!duplicated(nounFreqs$noun),]

data = merge(data, nounFreqs %>% rename(Noun = noun), by=c("Noun"), all.x=TRUE)

data = data %>% mutate(True_Minus_False.C = True_False_False-False_False_False-mean(True_False_False-False_False_False, na.rm=TRUE))

unique((data %>% filter(is.na(True_Minus_False.C)))$Noun)
# [1] conjecture  guess       insinuation intuition   observation

data$compatible.C = (grepl("_co", data$Condition)-0.5)
data$HasRC.C = (grepl("SCRC", data$Condition)-0.5)
data$HasSC.C = (0.5-grepl("NoSC", data$Condition))

data[data$HasSC.C < 0,]$compatible.C = 0
data[data$HasSC.C < 0,]$HasRC.C = 0


data$Item245 = grepl("245_", data$Item)


data = data %>% filter(!Item245)


library(brms)

for(pred in unique(data$predictability_weight)) {
  for(del in unique(data$deletion_rate)) {
    outpath = paste("posterior_summaries/analyze_M_VN3_lmer_effects_248_slopes_inter.R_", pred, "_", del, ".tsv", sep="")
    if(!file.exists(outpath)) {
    data2 = data %>% filter(predictability_weight == pred, deletion_rate == del)
#    if(del <= 0.4 | del >= 0.55) {
 #    if(del < 0.55 || TRUE) {
#if(pred == 0 || TRUE) {
    if(nrow(data2) > 0) {
          model2 = brm(SurprisalReweighted ~ True_Minus_False.C + HasRC.C + HasRC.C * True_Minus_False.C + True_Minus_False.C + compatible.C*HasRC.C + compatible.C*True_Minus_False.C + (1+True_Minus_False.C+compatible.C+HasRC.C+HasRC.C * True_Minus_False.C + True_Minus_False.C + compatible.C*HasRC.C + compatible.C*True_Minus_False.C|Item) + (1+compatible.C+HasRC.C+compatible.C*HasRC.C|Noun), data=data2 %>% filter(HasSC.C>0) %>% group_by(compatible.C,HasRC.C, True_Minus_False.C, Item, Noun) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), cores=4)
       write.table(summary(model2)$fixed, file=outpath, sep="\t")
    }
  }
}
}
#}
#}

#}
