
library(tidyr)
library(dplyr)
library(lme4)

data = read.csv("/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv/collectResults_Stims.py_VanDyke_Lewis_2003.tsv", sep="\t")

configs = unique(data %>% select(deletion_rate, predictability_weight))
#for(i in 1:nrow(configs)) {
#   model = (lmer(SurprisalReweighted ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|Item), data=data %>% filter(deletion_rate==configs$deletion_rate[[i]], predictability_weight==configs$predictability_weight[[i]]) %>% group_by(pp_rc, emb_c, someIntervention, Item, Condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) ))
#   print(configs[i,])
#   print(coef(summary(model)))
#}
##model = (lmer(SurprisalReweighted ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|item) + ( 1+ pp_rc + emb_c + someIntervention|Model), data=data %>% filter()) 


library(ggplot2)

#
## Limitation: The results are confounded with sentence position, which has a strong impact on GPT2 Surprisal
#plot = ggplot(data=data %>% group_by(Script, intervention, embedding, deletion_rate, predictability_weight) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=intervention, y=SurprisalReweighted, group=paste(Script,embedding), linetype=Script, color=embedding)) + geom_line() + facet_grid(predictability_weight ~ deletion_rate)
#ggsave(plot, file="cunnings-sturt_vanillaLSTM.pdf", height=10, width=10)
#
#

surprisalSmoothed = data.frame()

for(i in 1:nrow(configs)) {
   delta = configs$deletion_rate[[i]]
   lambda = configs$predictability_weight[[i]]
   surprisals = data %>% filter(abs(deletion_rate-delta)<=0.05, abs(predictability_weight-lambda)<=0.25) %>% group_by(Region, Condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) %>% mutate(deletion_rate=delta, predictability_weight=lambda)
   surprisalSmoothed = rbind(surprisalSmoothed, as.data.frame(surprisals))
}


plot = ggplot(data=surprisalSmoothed, aes(x=Region, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_line() + facet_grid(predictability_weight ~ deletion_rate)
ggsave(plot, file="vanDykeLewis2003_vanillaLSTM_smoothed.pdf", height=10, width=10)


surprisalSmoothed$Interference = ifelse(grepl("_1", surprisalSmoothed$Condition), "1 short", ifelse(grepl("_2", surprisalSmoothed$Condition), "3 high", "2 low"))
surprisalSmoothed$Ambiguous = grepl("nothat", surprisalSmoothed$Condition)
plot = ggplot(data=surprisalSmoothed %>% group_by(Condition, Interference, Ambiguous, predictability_weight, deletion_rate) %>% summarise(SurprisalReweighted=sum(SurprisalReweighted)), aes(x=Interference, y=SurprisalReweighted, group=Condition, fill=Ambiguous, color=Ambiguous)) + geom_bar(stat="identity", position="dodge") + facet_grid(predictability_weight ~ deletion_rate)
ggsave(plot, file="vanDykeLewis2003_vanillaLSTM_smoothed2.pdf", height=10, width=10)

# Interesting finding for Chen at al 2005: elevated surprisal emerges even at zero loss, in a by-item lmer strongly significant at region "critical_2"
