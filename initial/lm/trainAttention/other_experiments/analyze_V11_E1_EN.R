
library(tidyr)
library(dplyr)
library(lme4)

data = read.csv("/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv/collectResults_Stims.py_V11_E1_EN.tsv", sep="\t") %>% filter(Word == "the")

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

data$Condition = ifelse(grepl("Full", data$Condition), "Grammatical", "Ungrammatical")

surprisalSmoothed = data.frame()

for(i in 1:nrow(configs)) {
   delta = configs$deletion_rate[[i]]
   lambda = configs$predictability_weight[[i]]
   surprisals = data %>% filter(abs(deletion_rate-delta)<=0.05, abs(predictability_weight-lambda)<=0.25) %>% group_by(Condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) %>% mutate(deletion_rate=delta, predictability_weight=lambda)
   surprisalSmoothed = rbind(surprisalSmoothed, as.data.frame(surprisals))
}
# This comparison is actually not very interesting on its own

plot = ggplot(data=surprisalSmoothed, aes(x=Condition, y=SurprisalReweighted, color=Condition)) + geom_point() + facet_grid(predictability_weight ~ deletion_rate) + theme_bw() + theme(axis.text.x=element_blank (),  axis.ticks.x=element_blank ())
ggsave(plot, file="figures/V11_E1_EN_vanillaLSTM_smoothed.pdf", height=5, width=10)



