
library(tidyr)
library(dplyr)
library(lme4)

data = read.csv("/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv/collectResults_Stims.py_BartekEtal.tsv", sep="\t")

data$embedding = NA
data$intervention = NA

data[data$Condition == "a",]$embedding = "matrix"
data[data$Condition == "a",]$intervention = "none"

data[data$Condition == "b",]$embedding = "matrix"
data[data$Condition == "b",]$intervention = "pp"

data[data$Condition == "c",]$embedding = "matrix"
data[data$Condition == "c",]$intervention = "rc"

data[data$Condition == "d",]$embedding = "emb"
data[data$Condition == "d",]$intervention = "none"

data[data$Condition == "e",]$embedding = "emb"
data[data$Condition == "e",]$intervention = "pp"

data[data$Condition == "f",]$embedding = "emb"
data[data$Condition == "f",]$intervention = "rc"


data = data %>% mutate(pp_rc = case_when(intervention == "rc" ~ 1, intervention == "pp" ~ -1, TRUE ~ 0))
data = data %>% mutate(emb_c = case_when(embedding == "matrix" ~ -1, embedding == "emb" ~ 1))
data = data %>% mutate(someIntervention = case_when(intervention == "none" ~ -1, TRUE ~ 1))

configs = unique(data %>% select(deletion_rate, predictability_weight))
for(i in 1:nrow(configs)) {
   model = (lmer(SurprisalReweighted ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|Item), data=data %>% filter(deletion_rate==configs$deletion_rate[[i]], predictability_weight==configs$predictability_weight[[i]]) %>% group_by(pp_rc, emb_c, someIntervention, Item, Condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)) ))
   print(configs[i,])
   print(coef(summary(model)))
}
#model = (lmer(SurprisalReweighted ~ pp_rc * emb_c + someIntervention + (1 + pp_rc + emb_c + someIntervention|item) + ( 1+ pp_rc + emb_c + someIntervention|Model), data=data %>% filter()) 


library(ggplot2)


# Limitation: The results are confounded with sentence position, which has a strong impact on GPT2 Surprisal
plot = ggplot(data=data %>% group_by(intervention, embedding, deletion_rate, predictability_weight) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=intervention, y=SurprisalReweighted, group=embedding, color=embedding)) + geom_line() + facet_grid(predictability_weight ~ deletion_rate)
ggsave(plot, file="bartek_bb_vanillaLSTM.pdf", height=10, width=10)


