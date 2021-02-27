library(tidyr)
library(dplyr)



data = read.csv("averages_Agr2.tsv", quote='"', sep="\t")


counts = unique(read.csv("~/forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", sep="\t"))
counts$Ratio = log(counts$CountThat) - log(counts$CountBare)


scrc = read.csv("/home/user/forgetting/corpus_counts/wikipedia/RC_annotate/results/collectResults.py.tsv", sep="\t")
scrc = scrc %>% mutate(SC_Bias = (SC+1e-10)/(SC+RC+Other+3e-10))
scrc = scrc %>% mutate(Log_SC_Bias = log(SC_Bias))

counts = merge(counts, scrc, by=c("Noun")) %>% mutate(RatioSC = Ratio + Log_SC_Bias)



data = merge(data, counts, by=c("Noun"), all.x=TRUE)


library(ggplot2)


library(lme4)


dataM = data %>% group_by(Condition, deletion_rate, Noun) %>% summarise(Surprisal = mean(Surprisal), RatioSC=mean(RatioSC, na.rm=TRUE))
dataMJ = dataM


dataS = dataM %>% filter(Condition == "sing") %>% rename(Surprisal_S = Surprisal)
dataP = dataM %>% filter(Condition == "plur") %>% rename(Surprisal_P = Surprisal)

dataM = merge(dataS, dataP, by=c("Noun", "deletion_rate")) %>% mutate(SurpDiff = Surprisal_S-Surprisal_P)

cor.test(dataM$RatioSC.x, dataM$SurpDiff)
plot(dataM$RatioSC.x, dataM$SurpDiff)


dataM2 = dataM %>% group_by(Noun) %>% summarise(SurpDiff = mean(SurpDiff))
dataM2 = dataM2[order(dataM2$SurpDiff),]

dataM$Noun = factor(dataM$Noun, levels=dataM2$Noun)

plot = ggplot(dataM, aes(x=Noun, y=SurpDiff)) + geom_text(aes(label=Noun)) + facet_wrap(~deletion_rate)
write.csv(dataM2, file="output/analyze_Agr2.R")

dataMJ$Noun = factor(dataMJ$Noun, levels=dataM2$Noun)

plot = ggplot(dataMJ, aes(x=Noun, y=Surprisal, group=Condition, color=Condition)) + geom_text(aes(label=Noun)) + facet_wrap(~deletion_rate)


data = data %>% select(-Condition.C) %>% pivot_wider(names_from=c(Condition), values_from=c(Surprisal, ThatFraction))

data$SurpDiff = data$Surprisal_g-data$Surprisal_u
data$ThatDiff = data$ThatFraction_g-data$ThatFraction_u

######################################################
# This seems like a good version for visualizing it



dataGrid = dataNormJudg %>% filter(Region %in% c("V2_0", "V2_1", "V1_0","V1_1", "EOS_0")) %>% group_by(deletion_rate, predictability_weight, Noun, RatioSC, Condition, ID) %>% summarise(Surprisal = sum(Surprisal)) %>% mutate(SurprisalLogLikRatio = (Surprisal) - (ifelse(Condition == "u", 25.7, 43)))

tValues = data.frame()
for(deletion_rate_ in unique(dataNormJudg$deletion_rate)) {
  for(predictability_weight_ in unique(dataNormJudg$predictability_weight)) {
          d2 = dataGrid %>% filter(deletion_rate==deletion_rate_, predictability_weight==predictability_weight_)
	  d2$RatioSC.C = d2$RatioSC-mean(d2$RatioSC, na.rm=TRUE)
	  d2$Condition.C = (d2$Condition=="u")-0.5

          if(nrow(d2) > 1) {
                  if(length(unique(d2$ID)) > 1) {
          tValue = coef(summary(lmer(SurprisalLogLikRatio ~ RatioSC.C*Condition.C + (1+Condition.C|Noun) + (1+RatioSC+RatioSC.C*Condition.C+Condition.C|ID), data=d2 %>% mutate(ID=as.factor(ID)))))[4,3]
                  } else {
          tValue = coef(summary(lm(SurprisalLogLikRatio ~ RatioSC.C*Condition.C, data=d2 %>% mutate(ID=as.factor(ID)))))[4,3]
                  }
          tValues = rbind(tValues, data.frame(predictability_weight=c(predictability_weight_), deletion_rate=c(deletion_rate_), tValue=c(paste("t =",round(tValue, 4)))))
	  }
}
}


dataGrid = dataGrid %>% group_by(deletion_rate, predictability_weight, Noun, RatioSC, Condition) %>% summarise(SurprisalLogLikRatio = mean(SurprisalLogLikRatio))

dataGridNorm = dataGrid %>% group_by(deletion_rate, predictability_weight) %>% summarise(MeanSurprisalLogLikRatio = mean(SurprisalLogLikRatio), SDSurprisalLogLikRatio = sd(SurprisalLogLikRatio))
dataGrid = merge(dataGrid, dataGridNorm, by=c("deletion_rate", "predictability_weight"))
ggplot(dataGrid, aes(x=RatioSC, y=(SurprisalLogLikRatio - MeanSurprisalLogLikRatio)/SDSurprisalLogLikRatio, group=Condition, color=Condition)) + geom_smooth(method='lm') + theme_bw() + geom_point() + geom_label(data=tValues, aes(x=-2, y=0, label=tValue, group=NA, color=NA)) + facet_grid(deletion_rate~predictability_weight, scales="free") + xlab("log P(SC|the NOUN)") + ylab("Predicted Ratings")
ggsave("figures/logLikelihoodRatio_NormJudg_NEW_grid.pdf", height=10, width=6)

ggplot(dataGrid, aes(x=RatioSC, y=(SurprisalLogLikRatio - MeanSurprisalLogLikRatio)/SDSurprisalLogLikRatio, group=Condition, color=Condition)) + geom_smooth(method='lm') + theme_bw() + geom_point(alpha=0.2) + facet_grid(deletion_rate~predictability_weight, scales="free") + xlab("log P(SC|the NOUN)") + ylab("Predicted Ratings")
ggsave("figures/logLikelihoodRatio_NormJudg_NEW_grid_plain.pdf", height=5, width=5)



dataGrid$Condition_ = as.factor(1-as.numeric(dataGrid$Condition))
ggplot(dataGrid %>% filter(predictability_weight == 0.5, deletion_rate %in% c(0.1, 0.3, 0.5, 0.7)), aes(x=RatioSC, y=(SurprisalLogLikRatio - MeanSurprisalLogLikRatio)/SDSurprisalLogLikRatio, group=Condition_, color=Condition_)) + geom_smooth(method='lm') + theme_bw() + geom_point(alpha=0.2) + facet_grid(deletion_rate~1, scales="free") + xlab("log P(SC|the NOUN)") + ylab("Predicted Ratings")



