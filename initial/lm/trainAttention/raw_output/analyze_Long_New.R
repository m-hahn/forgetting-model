

data = read.csv("averages_Long.tsv", quote='"', sep="\t")



library(ggplot2)

counts = read.csv("../../../../../forgetting/fromCorpus_counts.csv", sep="\t")
counts$Ratio = counts$True_False - counts$False_False



data = merge(data, counts, by=c("Noun"), all.x=TRUE)

library(lme4)

data$Condition.C = ifelse(data$Condition == "u", -0.5, 0.5)

library(tidyr)
library(dplyr)
summary(lmer(Surprisal ~ Condition.C*Ratio + (1+Condition.C|Noun), data=data %>% filter(Region == "EOS")))
summary(lmer(Surprisal ~ Condition.C*Ratio + (1+Condition.C|Noun), data=data %>% filter(Region == "V1")))



ggplot(data %>% group_by(Region, Condition) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Region, y=Surprisal, group=Condition, color=Condition)) + geom_line()

dataLong = data


plot = ggplot(data=dataLong %>% filter(Region == "EOS") %>% group_by(Condition, predictability_weight, deletion_rate) %>% summarise(ThatFraction = mean(ThatFraction)), aes(x=predictability_weight, y=deletion_rate)) + geom_tile(aes(fill=ThatFraction))
plot = plot +  scale_fill_gradient2() + facet_wrap(~Condition)


data = data %>% select(-Condition.C) %>% pivot_wider(names_from=c(Condition), values_from=c(Surprisal, ThatFraction))

data$SurpDiff = data$Surprisal_g-data$Surprisal_u
data$ThatDiff = data$ThatFraction_g-data$ThatFraction_u

dataEOS = data %>% filter(Region == "EOS")
dataV1 = data %>% filter(Region == "V1")





correlations_EOS_ByID = dataEOS %>% group_by(ID, predictability_weight, deletion_rate) %>% summarise(Correlation = cor(SurpDiff, Ratio), Correlation_That = cor(ThatDiff, Ratio))
correlations_V1_ByID = dataV1 %>% group_by(ID, predictability_weight, deletion_rate) %>% summarise(Correlation = cor(SurpDiff, Ratio), Correlation_That = cor(ThatDiff, Ratio))


correlations_EOS = dataEOS %>% group_by(predictability_weight, deletion_rate) %>% summarise(Correlation = cor(SurpDiff, Ratio), Correlation_That = cor(ThatDiff, Ratio))
correlations_V1 = dataV1 %>% group_by(predictability_weight, deletion_rate) %>% summarise(Correlation = cor(SurpDiff, Ratio), Correlation_That = cor(ThatDiff, Ratio))

plot = ggplot(data=correlations_EOS, aes(x=predictability_weight, y=deletion_rate)) + geom_tile(aes(fill=Correlation))
plot = plot +  scale_fill_gradient2() #+ facet_wrap(~denoiser)

plot = ggplot(data=correlations_V1, aes(x=predictability_weight, y=deletion_rate)) + geom_tile(aes(fill=Correlation))
plot = plot +  scale_fill_gradient2() #+ facet_wrap(~denoiser)

plot = ggplot(data=correlations_EOS, aes(x=predictability_weight, y=deletion_rate)) + geom_tile(aes(fill=Correlation_That))
plot = plot +  scale_fill_gradient2() #+ facet_wrap(~denoiser)

ggplot(dataLong %>% filter(Region == "EOS", deletion_rate == 0.3) %>% group_by(Noun, Ratio, Condition) %>% summarise(Surprisal = mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth() + geom_point() + geom_label(aes(label=Noun))

ggplot(dataLong %>% filter(Region == "EOS", deletion_rate == 0.3) %>% group_by(Noun, Ratio, Condition) %>% summarise(ThatFraction = mean(ThatFraction)), aes(x=Ratio, y=ThatFraction, group=Condition, color=Condition)) + geom_smooth() + geom_point() + geom_label(aes(label=Noun))



ggplot(dataLong %>% filter(Region == "V1", deletion_rate == 0.3) %>% group_by(Noun, Ratio, Condition) %>% summarise(Surprisal = mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth() + geom_point() + geom_label(aes(label=Noun))

ggplot(dataLong %>% filter(Region == "V1", deletion_rate == 0.3) %>% group_by(Noun, Ratio, Condition) %>% summarise(ThatFraction = mean(ThatFraction)), aes(x=Ratio, y=ThatFraction, group=Condition, color=Condition)) + geom_smooth() + geom_point() + geom_label(aes(label=Noun))



ggplot(dataLong %>% filter(Region %in% c("V1","EOS"), deletion_rate == 0.3) %>% group_by(Noun, Ratio, Condition) %>% summarise(Surprisal = mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth() + geom_point() + geom_label(aes(label=Noun))

ggplot(dataLong %>% filter(Region %in% c("V2"), deletion_rate == 0.3) %>% group_by(Noun, Ratio, Condition) %>% summarise(Surprisal = mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth() + geom_point() + geom_label(aes(label=Noun))


ggplot(dataLong %>% filter(Region %in% c("V2", "V1","EOS"), deletion_rate == 0.3) %>% group_by(Noun, Ratio, Condition, ID) %>% summarise(Surprisal = sum(Surprisal)) %>% group_by(Noun, Ratio, Condition) %>% summarise(Surprisal = mean(Surprisal - (ifelse(Condition == "u", 25.7, 43)))), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method='lm') + geom_point() + geom_label(aes(label=Noun))

######################################################
# This seems like a good version for visualizing it
ggplot(dataLong %>% filter(Region %in% c("V2", "V1","EOS")) %>% group_by(deletion_rate, Noun, Ratio, Condition, ID) %>% summarise(Surprisal = sum(Surprisal)) %>% group_by(deletion_rate, Noun, Ratio, Condition) %>% summarise(SurprisalLogLikRatio = mean(Surprisal) - mean(ifelse(Condition == "u", 25.7, 43))), aes(x=Ratio, y=SurprisalLogLikRatio, group=Condition, color=Condition)) + geom_smooth(method='lm') + geom_point() + facet_wrap(~deletion_rate, scales="free")
ggsave("figures/logLikelihoodRatio_NEW_byDeletionRate.pdf")

ggplot(dataLong %>% filter(Region %in% c("V2", "V1","EOS")) %>% group_by(predWeight, Noun, Ratio, Condition, ID) %>% summarise(Surprisal = sum(Surprisal)) %>% group_by(predWeight, Noun, Ratio, Condition) %>% summarise(SurprisalLogLikRatio = mean(Surprisal) - mean(ifelse(Condition == "u", 25.7, 43))), aes(x=Ratio, y=SurprisalLogLikRatio, group=Condition, color=Condition)) + geom_smooth(method='lm') + geom_point() + facet_wrap(~predWeight, scales="free")
ggsave("figures/logLikelihoodRatio_NEW_byPredWeight.pdf")

dataGrid = dataLong %>% filter(Region %in% c("V2", "V1","EOS")) %>% group_by(deletion_rate, predictability_weight, Noun, Ratio, Condition, ID) %>% summarise(Surprisal = sum(Surprisal)) %>% group_by(deletion_rate, predictability_weight, Noun, Ratio, Condition) %>% summarise(SurprisalLogLikRatio = mean(Surprisal) - mean(ifelse(Condition == "u", 25.7, 43)))
dataGridNorm = dataGrid %>% group_by(deletion_rate, predictability_weight) %>% summarise(MeanSurprisalLogLikRatio = mean(SurprisalLogLikRatio), SDSurprisalLogLikRatio = sd(SurprisalLogLikRatio))
dataGrid = merge(dataGrid, dataGridNorm, by=c("deletion_rate", "predictability_weight"))
ggplot(dataGrid, aes(x=Ratio, y=(SurprisalLogLikRatio - MeanSurprisalLogLikRatio)/SDSurprisalLogLikRatio, group=Condition, color=Condition)) + geom_smooth(method='lm') + geom_point() + facet_grid(deletion_rate~predictability_weight, scales="free")
ggsave("figures/logLikelihoodRatio_NEW_grid.pdf", height=10, width=6)




