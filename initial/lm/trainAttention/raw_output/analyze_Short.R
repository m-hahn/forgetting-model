library(tidyr)
library(dplyr)



data = read.csv("averages_Short.tsv", quote='"', sep="\t")


counts = unique(read.csv("~/forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", sep="\t"))
counts$Ratio = log(counts$CountThat) - log(counts$CountBare)


scrc = read.csv("/home/user/forgetting/corpus_counts/wikipedia/RC_annotate/results/collectResults.py.tsv", sep="\t")
scrc = scrc %>% mutate(SC_Bias = (SC+1e-10)/(SC+RC+Other+3e-10))
scrc = scrc %>% mutate(Log_SC_Bias = log(SC_Bias))

counts = merge(counts, scrc, by=c("Noun")) %>% mutate(RatioSC = Ratio + Log_SC_Bias)



data = merge(data, counts, by=c("Noun"), all.x=TRUE)


library(ggplot2)


library(lme4)


dataM = data %>% group_by(ID, ScriptName, Condition, deletion_rate, predictability_weight, Noun) %>% summarise(Surprisal = mean(Surprisal), Ratio=mean(Ratio, na.rm=TRUE), RatioSC=mean(RatioSC, na.rm=TRUE), ThatFraction=mean(ThatFraction, na.rm=TRUE))

plot = ggplot(dataM %>% group_by(deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_wrap(~deletion_rate)
plot = ggplot(dataM %>% group_by(deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth() + geom_text(aes(label=Noun)) + facet_wrap(~deletion_rate)

plot = ggplot(dataM %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth() + geom_text(aes(label=Noun)) + facet_grid(predictability_weight~deletion_rate)
plot = ggplot(dataM %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_grid(predictability_weight~deletion_rate)
plot = ggplot(dataM %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + facet_grid(predictability_weight~deletion_rate)






tValues = data.frame()
for(deletion_rate_ in unique(dataM$deletion_rate)) {
  for(predictability_weight_ in unique(dataM$predictability_weight)) {
          d2 = dataM %>% filter(deletion_rate==deletion_rate_, predictability_weight==predictability_weight_) %>% group_by()
	  d2$Ratio.C = d2$Ratio-mean(d2$Ratio, na.rm=TRUE)
	  d2$Condition.C = (d2$Condition=="SC")-0.5

          if(nrow(d2) > 1) {
                  if(length(unique(d2$ID)) > 1) {
          tValue = coef(summary(lmer(Surprisal ~ Ratio.C*Condition.C + (1+Condition.C|Noun) + (1+Ratio+Ratio.C*Condition.C+Condition.C|ID), data=d2 %>% mutate(ID=as.factor(ID)))))[4,3]
                  } else {
          tValue = coef(summary(lm(Surprisal ~ Ratio.C*Condition.C, data=d2 %>% mutate(ID=as.factor(ID)))))[4,3]
                  }
          tValues = rbind(tValues, data.frame(predictability_weight=c(predictability_weight_), deletion_rate=c(deletion_rate_), tValue=c(paste("t =",round(tValue, 4)))))
	  }
  }
}

plot = ggplot(dataM %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + facet_grid(predictability_weight~deletion_rate) + geom_label(data=tValues, aes(x=-2, y=0, label=tValue, group=NA, color=NA))







u = dataM %>% filter(Condition == "SC") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)) %>% group_by(predictability_weight, deletion_rate) %>% summarise(corr = cor(Surprisal, Ratio))

u1 = dataM %>% filter(Condition == "SC")
u2 = dataM %>% filter(Condition == "NoSC")
u = merge(u1, u2, by=c("ID", "Noun")) %>% mutate(SurpDiff = Surprisal.x-Surprisal.y) %>% group_by(Noun) %>% summarise(SurpDiff=mean(SurpDiff), Surprisal_NoSC = mean(Surprisal.y), Surprisal_SC=mean(Surprisal.x))

write.csv( u[order(u$SurpDiff),], file="output/analyze_Short.R.tsv")

plot = ggplot(dataM %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFraction=mean(ThatFraction)), aes(x=Ratio, y=ThatFraction, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + facet_grid(predictability_weight~deletion_rate)
plot = ggplot(dataM %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFraction=mean(ThatFraction)), aes(x=exp(Ratio), y=ThatFraction, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + facet_grid(predictability_weight~deletion_rate)

##############################
# OLD

plot = ggplot(dataM, aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_wrap(~deletion_rate+ScriptName+ID)

plot = ggplot(dataM, aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_wrap(~deletion_rate+ScriptName+ID)


plot = ggplot(dataM %>% filter(Condition == "SC"), aes(x=Ratio, y=ThatFraction, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_wrap(~deletion_rate+ScriptName+ID)

plot = ggplot(dataM %>% filter(ScriptName == "char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_NoLM_NoPos", Condition == "SC"), aes(x=Ratio, y=ThatFraction, group=Condition, color=Condition)) + geom_text(aes(label=Noun)) + geom_smooth(method="lm") + facet_wrap(~deletion_rate+ScriptName+ID)
plot = ggplot(dataM %>% filter(ScriptName == "char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_NoLM_NoPos", Condition == "SC"), aes(x=Ratio, y=ThatFraction, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_wrap(~deletion_rate+ScriptName+ID)
plot = ggplot(dataM %>% filter(ScriptName == "char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Stims", Condition == "SC"), aes(x=Ratio, y=ThatFraction, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_wrap(~deletion_rate+ScriptName+ID)
plot = ggplot(dataM %>% group_by(deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFraction=mean(ThatFraction)), aes(x=Ratio, y=ThatFraction, group=Condition, color=Condition)) + geom_smooth() + geom_text(aes(label=Noun)) + facet_wrap(~deletion_rate)



dataS = dataM %>% filter(Condition == "SC") %>% rename(Surprisal_S = Surprisal)
dataP = dataM %>% filter(Condition == "NoSC") %>% rename(Surprisal_P = Surprisal)

dataM = merge(dataS, dataP, by=c("Noun", "deletion_rate")) %>% mutate(SurpDiff = Surprisal_S-Surprisal_P)

cor.test(dataM$RatioSC.x, dataM$SurpDiff)
plot(dataM$RatioSC.x, dataM$SurpDiff)


dataM2 = dataM %>% group_by(Noun) %>% summarise(SurpDiff = mean(SurpDiff), Surprisal_S=mean(Surprisal_S), Surprisal_P=mean(Surprisal_P))
dataM2 = dataM2[order(dataM2$SurpDiff),]

dataM$Noun = factor(dataM$Noun, levels=dataM2$Noun)

plot = ggplot(dataM, aes(x=Noun, y=SurpDiff)) + geom_text(aes(label=Noun)) + facet_wrap(~deletion_rate)
#write.csv(dataM2, file="output/analyze_Short.R")

plot = ggplot(dataM, aes(x=RatioSC.x, y=SurpDiff)) + geom_text(aes(label=Noun)) + facet_wrap(~deletion_rate)


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



