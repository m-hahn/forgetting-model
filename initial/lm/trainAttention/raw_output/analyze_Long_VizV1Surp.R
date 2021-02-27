

data = read.csv("averages_Long.tsv", quote='"', sep="\t")



library(ggplot2)

counts = read.csv("../../../../../forgetting/fromCorpus_counts.csv", sep="\t")
counts$Ratio = counts$True_False - counts$False_False



data = merge(data, counts, by=c("Noun"), all.x=TRUE)

library(lme4)

data$Condition.C = ifelse(data$Condition == "u", -0.5, 0.5)

library(tidyr)
library(dplyr)

dataLong = data

data = data %>% select(-Condition.C) %>% pivot_wider(names_from=c(Condition), values_from=c(Surprisal, ThatFraction))



tValues = data.frame()
for(deletion_rate_ in unique(dataLong$deletion_rate)) {
  for(predictability_weight_ in unique(dataLong$predictability_weight)) {
	  d2 = dataLong %>% filter(deletion_rate==deletion_rate_, predictability_weight==predictability_weight_)
	  if(nrow(d2) > 1) {
		  if(length(unique(d2$ID)) > 1) {
          tValue = coef(summary(lmer(Surprisal ~ Ratio + (1|Noun) + (1+Ratio|ID), data=d2 %>% filter(Region == "V1") %>% mutate(ID=as.factor(ID)))))[2,3]
		  } else {
          tValue = coef(summary(lm(Surprisal ~ Ratio, data=d2 %>% filter(Region == "V1") %>% mutate(ID=as.factor(ID)))))[2,3]
		  }
          tValues = rbind(tValues, data.frame(predictability_weight=c(predictability_weight_), deletion_rate=c(deletion_rate_), tValue=c(paste("t =",round(tValue, 4))), yPosition=c(max((d2 %>% filter(Region == "V1") %>%  group_by(Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)))$Surprisal))))
	  }
}
}

dataV1 = dataLong %>% filter(Region == "V1")  %>% group_by(Noun, Ratio, predictability_weight, deletion_rate) %>% summarise(Surprisal=mean(Surprisal))
ggplot(dataV1, aes(x=Ratio, y=Surprisal)) + geom_smooth(method="lm") + geom_point()  + geom_text(data=tValues, aes(x=-2, y=yPosition, label=tValue), inherit.aes=FALSE, parse=FALSE)+ facet_grid(deletion_rate~predictability_weight, scales="free") + theme_bw()
ggsave("figures/surprisals_OnlyGramm_V1_grid.pdf", height=10, width=6)


