library(dplyr)
library(tidyr)

data = read.csv("/u/scr/mhahn/reinforce-logs-both-short/stimuli-full-logs-tsv-EIS/collectResults_Stims_ErrorIdentification.py_ChristiansonGardenpathLinger_Model.tsv", sep="\t")


library(lme4)

#
#
#library(ggplot2)
#plot = ggplot(data %>%filter(Script == "errorIdentification_Erasure3_NoSanity.py") %>% group_by(deletion_rate, predictability_weight, Region, Word) %>% summarise(EISReweighted=mean(EISReweighted), SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Region, y=EISReweighted, color=Word)) + geom_text(aes(label=Word)) + facet_grid(predictability_weight~deletion_rate)
#ggsave(plot, file="figures/analyze_ErrorIdentification_Model_R_ToyLinger.pdf", height=15, width=40)
#plot = ggplot(data %>%filter(Script == "errorIdentification_Erasure3_NoSanity.py") %>% group_by(deletion_rate, predictability_weight, Region, Word) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted), SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Region, y=SurprisalReweighted, color=Word)) + geom_text(aes(label=Word)) + facet_grid(predictability_weight~deletion_rate)
#ggsave(plot, file="figures/analyze_ErrorIdentification_Model_R_Toy_SurprisalLinger.pdf", height=15, width=40)
#

#model = (lmer(EISReweighted ~ Word + (1|ID), data=data%>%filter(Script == "errorIdentification_Erasure3_NoSanity.py", Word %in% c("tossed", "thrown"))))
#print(summary(model))
#
#model = (lmer(SurprisalReweighted ~ Word + (1|ID), data=data%>%filter(Script == "errorIdentification_Erasure3_NoSanity.py", Word %in% c("tossed", "thrown"))))
#print(summary(model))


library(ggplot2)


data$Plausible = grepl("_plaus" , data$Condition) - 0.5
data$Short = grepl("_short" , data$Condition) - 0.5
data$GardenPath = (!grepl("nogp" , data$Condition)) - 0.5

#model = (lmer(EISReweighted ~ Plausible + Short + GardenPath  + (1|ID) + (1|Item), data=data%>%filter(Script == "errorIdentification_Erasure4.py", Region %in% c("fifth_0", "sixth_0"))))
#print(summary(model))
#model = (lmer(SurprisalReweighted ~ Ambiguous * Short + (1|ID) + (1|Item), data=data%>%filter(Script == "errorIdentification_Erasure4.py", Region %in% c("participle_0"))))
#print(summary(model))



data2 = data%>%filter(Script == "errorIdentification_Erasure4.py", Region %in% c("fifth_0","sixth_0")) %>% group_by(deletion_rate, predictability_weight, Item, Condition, Plausible, Short, GardenPath) %>% summarise(EISReweighted=mean(EISReweighted), SurprisalReweighted=mean(SurprisalReweighted))

sink("output/analyze_ErrorIdentification_Model.RLinger.tsv")
cat("delta", "lambda", "plausible", "short", "gardenPath", "short_gardenPath")
config = unique(data2 %>% group_by() %>% select(deletion_rate, predictability_weight))
for(i in 1:nrow(config)) {
  delta = config$deletion_rate[[i]]
  lambda = config$predictability_weight[[i]]
  u = data2 %>% filter(deletion_rate==delta, predictability_weight==lambda)
  model = (lmer(EISReweighted ~ Plausible + Short + GardenPath * Short +  (1|Item), data=u))
  cat(delta, lambda, summary(model)$coef[[2,3]], summary(model)$coef[[3,3]], summary(model)$coef[[4,3]], summary(model)$coef[[5,3]], "\n", sep="\t")
}
sink()

plot = ggplot(data2 %>% group_by(deletion_rate, predictability_weight, Condition) %>% summarise(EISReweighted=mean(EISReweighted)), aes(x=Condition, y=EISReweighted, color=Condition)) + geom_point() + facet_grid(predictability_weight~deletion_rate)
ggsave(plot, file="figures/analyze_ErrorIdentification_Model_R_Tabor_EISLinger.pdf", height=10, width=15)
plot = ggplot(data2 %>% group_by(deletion_rate, predictability_weight, Condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Condition, y=SurprisalReweighted, color=Condition)) + geom_point() + facet_grid(predictability_weight~deletion_rate)
ggsave(plot, file="figures/analyze_ErrorIdentification_Model_R_Tabor_SurprisalLinger.pdf", height=10, width=15)



#data2 = data%>%filter(Script == "errorIdentification_Erasure4.py", Region %in% c("fifth_0","sixth_0")) %>% group_by(deletion_rate, predictability_weight, Item, Ambiguous, Short) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))
#for(i in 1:nrow(config)) {
#  delta = config$deletion_rate[[i]]
#  lambda = config$predictability_weight[[i]]
#  u = data2 %>% filter(deletion_rate==delta, predictability_weight==lambda)
#  model = (lmer(SurprisalReweighted ~ Ambiguous * Short +  (1|Item), data=u))
#  cat("SURP", delta, lambda, summary(model)$coef[[2,3]], summary(model)$coef[[3,3]], summary(model)$coef[[4,3]], "\n", sep="\t")
#}
#
