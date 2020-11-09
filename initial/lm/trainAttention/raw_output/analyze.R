

data = read.csv("averages.tsv", quote='"', sep="\t")



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


#{'V2': -18.765784864784592, 'V1': -25.713731532729923}
ggplot(dataLong %>% group_by(ID, Noun, Ratio, Condition) %>% summarise(Surprisal = sum(Surprisal)) %>% mutate(Surprisal = Surprisal - 25.7 - ifelse(Condition == "u", 0, 18.7)) %>% group_by(Noun, Ratio, Condition) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + geom_label(aes(label=Noun))

# RATINGS
ggplot(dataLong %>% group_by(ID, Noun, Ratio, Condition, deletion_rate) %>% summarise(Surprisal = sum(Surprisal)) %>% mutate(Surprisal = Surprisal - 25.7 - ifelse(Condition == "u", 0, 18.7)) %>% group_by(Noun, Ratio, Condition, deletion_rate) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + facet_wrap(~deletion_rate, scales="free")

tValues = data.frame()
for(deletion_rate_ in unique(dataLong$deletion_rate)) {
   tValue = coef(summary(lmer(Surprisal ~ Condition.C*Ratio + (1+Condition.C|Noun) + (1+Condition.C+Ratio|ID), data=dataLong %>% filter(deletion_rate==deletion_rate_) %>% mutate(ID=as.factor(ID)) %>% group_by(Noun, ID, Condition, Condition.C, Noun, Ratio) %>% summarise(Surprisal = sum(Surprisal)) %>% mutate(Surprisal = Surprisal - 25.7 - ifelse(Condition == "u", 0, 18.7)))))[4,3]
   tValues = rbind(tValues, data.frame(deletion_rate=c(deletion_rate_), tValue=c(paste("t =",round(tValue, 4))), yPosition=c(max(((dataLong %>% filter(deletion_rate==deletion_rate_) %>% mutate(ID=as.factor(ID)) %>% group_by(Noun, ID, Condition, Condition.C, Noun, Ratio) %>% summarise(Surprisal = sum(Surprisal)) %>% mutate(Surprisal = Surprisal - 25.7 - ifelse(Condition == "u", 0, 18.7)) %>% summarise(Surprisal=mean(Surprisal)))$Surprisal)))))
}




ggplot(dataLong %>% group_by(ID, Noun, Ratio, Condition, deletion_rate) %>% summarise(Surprisal = sum(Surprisal)) %>% mutate(Surprisal = Surprisal - 25.7 - ifelse(Condition == "u", 0, 18.7)) %>% group_by(Noun, Ratio, Condition, deletion_rate) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + facet_wrap(~deletion_rate, scales="free") + geom_text(data=tValues, aes(x=-2, y=yPosition, label=tValue), inherit.aes=FALSE, parse=FALSE)
ggsave("figures/logLikelihoodRatio_byDeletionRate.pdf")


# Reading Times
ggplot(dataLong %>% filter(Region == "V1")  %>% group_by(Noun, Ratio, Condition, deletion_rate) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + facet_wrap(~deletion_rate, scales="free")

tValues = data.frame()
for(deletion_rate_ in unique(dataLong$deletion_rate)) {
   tValue = coef(summary(lmer(Surprisal ~ Condition.C*Ratio + (1+Condition.C|Noun) + (1+Condition.C+Ratio|ID), data=dataLong %>% filter(Region == "V1", deletion_rate==deletion_rate_) %>% mutate(ID=as.factor(ID)))))[4,3]
   tValues = rbind(tValues, data.frame(deletion_rate=c(deletion_rate_), tValue=c(paste("t =",round(tValue, 4))), yPosition=c(max((dataLong %>% filter(Region == "V1", deletion_rate==deletion_rate_) %>%  group_by(Noun, Ratio, Condition, deletion_rate) %>% summarise(Surprisal=mean(Surprisal)))$Surprisal))))
}

ggplot(dataLong %>% filter(Region == "V1")  %>% group_by(Noun, Ratio, Condition, deletion_rate) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point()  + geom_text(data=tValues, aes(x=-2, y=yPosition, label=tValue), inherit.aes=FALSE, parse=FALSE)+ facet_wrap(~deletion_rate, scales="free")
ggsave("figures/surprisals_V1_byDeletionRate.pdf")


# Reading Times only in Grammatical Sentences
ggplot(dataLong %>% filter(Region == "V1", Condition=="g")  %>% group_by(Noun, Ratio, deletion_rate) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal)) + geom_smooth(method="lm") + geom_point() + facet_wrap(~deletion_rate, scales="free")

tValues = data.frame()
for(deletion_rate_ in unique(dataLong$deletion_rate)) {
   tValue = coef(summary(lmer(Surprisal ~ Ratio + (1|Noun) + (1+Ratio|ID), data=dataLong %>% filter(Region == "V1", deletion_rate==deletion_rate_) %>% mutate(ID=as.factor(ID)))))[2,3]
   tValues = rbind(tValues, data.frame(deletion_rate=c(deletion_rate_), tValue=c(paste("t =",round(tValue, 4))), yPosition=c(max((dataLong %>% filter(Region == "V1", deletion_rate==deletion_rate_) %>%  group_by(Noun, Ratio, deletion_rate) %>% summarise(Surprisal=mean(Surprisal)))$Surprisal))))
}

ggplot(dataLong %>% filter(Region == "V1")  %>% group_by(Noun, Ratio, deletion_rate) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal)) + geom_smooth(method="lm") + geom_point()  + geom_text(data=tValues, aes(x=-2, y=yPosition, label=tValue), inherit.aes=FALSE, parse=FALSE)+ facet_wrap(~deletion_rate, scales="free")
ggsave("figures/surprisals_OnlyGramm_V1_byDeletionRate.pdf")




ggplot(dataLong %>% filter(!(Region == "V2" & Condition == "u")) %>% group_by(deletion_rate, Noun, Ratio, Condition) %>% summarise(Surprisal = mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + facet_wrap(~deletion_rate, scales="free")

###################3
ggplot(dataLong %>% filter(Region == "V1") %>% group_by(deletion_rate, Noun, Ratio, Condition) %>% summarise(Surprisal = mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + facet_wrap(~deletion_rate, scales="free")
####################

ggplot(dataLong %>% filter(Region != "V2") %>% group_by(deletion_rate, Noun, Ratio, Condition) %>% summarise(Surprisal = mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + facet_wrap(~deletion_rate, scales="free")

ggplot(dataLong %>% filter(Region == "V1") %>% group_by(predictability_weight, deletion_rate, Noun, Ratio, Condition) %>% summarise(Surprisal = mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + facet_wrap(~predictability_weight+deletion_rate, scales="free")




		  #with open(", "r") as inFile:                                                                                                                                   
#   counts = [x.split("\t") for x in inFile.read().strip().split("\n")]                                                                                                                                      
#   header = counts[0]                                                                                                                                                                                       
#   header = dict(list(zip(header, range(len(header)))))                                                                                                                                                     
#   counts = {line[0] : line[1:] for line in counts}                                                                                                                                                         
#                                                                                                                                                                                                            
#topNouns = [x for x in topNouns if x in counts]                                                                                                                                                             
#topNouns = sorted(list(set(topNouns)), key=lambda x:float(counts[x][header["True_False"]])-float(counts[x][header["False_False"]]))         


