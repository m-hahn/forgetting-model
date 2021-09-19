
library(tidyr)
library(dplyr)
library(lme4)

data_GG = read.csv("/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv/collectResults_Stims.py_BartekGG.tsv", sep="\t") %>% mutate(Stimuli = "Grodner & Gibson")
data_B = read.csv("/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv/collectResults_Stims.py_BartekEtal.tsv", sep="\t") %>% mutate(Stimuli = "Bartek et al")

data = rbind(data_B, data_GG)

data$Embedding = NA
data$Intervening = NA

data[data$Condition == "a",]$Embedding = "Matrix"
data[data$Condition == "a",]$Intervening = "none"

data[data$Condition == "b",]$Embedding = "Matrix"
data[data$Condition == "b",]$Intervening = "pp"

data[data$Condition == "c",]$Embedding = "Matrix"
data[data$Condition == "c",]$Intervening = "rc"

data[data$Condition == "d",]$Embedding = "Embedded"
data[data$Condition == "d",]$Intervening = "none"

data[data$Condition == "e",]$Embedding = "Embedded"
data[data$Condition == "e",]$Intervening = "pp"

data[data$Condition == "f",]$Embedding = "Embedded"
data[data$Condition == "f",]$Intervening = "rc"


data = data %>% mutate(pp_rc = case_when(Intervening == "rc" ~ 1, Intervening == "pp" ~ -1, TRUE ~ 0))
data = data %>% mutate(emb_c = case_when(Embedding == "Matrix" ~ -1, Embedding == "Embedded" ~ 1))
data = data %>% mutate(someIntervention = case_when(Intervening == "none" ~ -1, TRUE ~ 1))

configs = unique(data %>% select(deletion_rate, predictability_weight))
#for(i in 1:nrow(configs)) {
#   model = (lmer(SurprisalReweighted ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|Item), data=data %>% filter(deletion_rate==configs$deletion_rate[[i]], predictability_weight==configs$predictability_weight[[i]]) %>% group_by(pp_rc, emb_c, someIntervention, Item, Condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) ))
#   print(configs[i,])
#   print(coef(summary(model)))
#}
#model = (lmer(SurprisalReweighted ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|item) + ( 1+ pp_rc + emb_c + someIntervention|Model), data=data %>% filter()) 


library(ggplot2)


# Limitation: The results are confounded with sentence position, which has a strong impact on GPT2 Surprisal
plot = ggplot(data=data %>% group_by(Script, Intervening, Embedding, deletion_rate, predictability_weight) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Intervening, y=SurprisalReweighted, group=paste(Script,Embedding), linetype=Script, color=Embedding)) + geom_line() + facet_grid(predictability_weight ~ deletion_rate)
ggsave(plot, file="bartek_joint_vanillaLSTM.pdf", height=10, width=10)



surprisalSmoothed = data.frame()

for(i in 1:nrow(configs)) {
   delta = configs$deletion_rate[[i]]
   lambda = configs$predictability_weight[[i]]
   surprisals = data %>% filter(abs(deletion_rate-delta)<=0.05, abs(predictability_weight-lambda)<=0.25) %>% group_by(Stimuli, Intervening, Embedding) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) %>% mutate(deletion_rate=delta, predictability_weight=lambda)
   surprisalSmoothed = rbind(surprisalSmoothed, as.data.frame(surprisals))
}


plot = ggplot(data=surprisalSmoothed %>% mutate(EmbeddingID = ifelse(Embedding == "Matrix", 1, 2)) %>% group_by(Stimuli, Intervening, EmbeddingID,Embedding, deletion_rate, predictability_weight) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=paste(EmbeddingID, Intervening), y=SurprisalReweighted, group=paste(EmbeddingID, Stimuli), color=Stimuli)) + theme_bw() + geom_line() + facet_grid(predictability_weight ~ deletion_rate) + ylab("Model Surprisal") + theme(legend.position = 'bottom')
ggsave(plot, file="figures/bartek_joint_vanillaLSTM_smoothed.pdf", height=5, width=15)


human = read.csv("analyzeBartek_human.tsv", sep="\t") %>% mutate(EmbeddingID = ifelse(Embedding == "Matrix", 1, 2))
#plot = ggplot(data=human, aes(x=Intervening, y=ReadingTime, group=paste(Embedding), color=Embedding)) + theme_bw() + geom_line() + facet_grid(~Measure) + ylab("Reading Time")  + theme(legend.position = 'bottom')
plot = ggplot(data=human, aes(x=paste(EmbeddingID, Intervening), y=ReadingTime, group=paste(EmbeddingID, Stimuli), color=Stimuli)) + theme_bw() + geom_line() + facet_grid(~Measure) + ylab("Reading Time") + theme(legend.position = 'bottom') + geom_errorbar(aes(ymin=ReadingTime-Error*0.3, ymax=ReadingTime+Error*0.3))
ggsave(plot, file="figures/bartek_joint_human.pdf", height=3, width=7)
#
