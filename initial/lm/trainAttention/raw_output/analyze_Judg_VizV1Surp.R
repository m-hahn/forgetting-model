

data = read.csv("averages_NormJudg.tsv", quote='"', sep="\t")



library(ggplot2)

#counts = read.csv("../../../../../forgetting/fromCorpus_counts.csv", sep="\t")
#counts$Ratio = counts$True_False - counts$False_False

counts = unique(read.csv("~/forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", sep="\t"))
counts$Ratio = log(counts$CountThat) - log(counts$CountBare)


scrc = read.csv("/home/user/forgetting/corpus_counts/wikipedia/RC_annotate/results/collectResults.py.tsv", sep="\t")
scrc = scrc %>% mutate(SC_Bias = (SC+1e-10)/(SC+RC+Other+3e-10))
scrc = scrc %>% mutate(Log_SC_Bias = log(SC_Bias))

counts = merge(counts, scrc, by=c("Noun")) %>% mutate(RatioSC = Ratio + Log_SC_Bias)



data = merge(data, counts, by=c("Noun"), all.x=TRUE)

library(lme4)

data$Condition.C = ifelse(data$Condition == "u", -0.5, 0.5)

library(tidyr)
library(dplyr)

dataNormJudg = data

data = data %>% select(-Condition.C) %>% pivot_wider(names_from=c(Condition), values_from=c(Surprisal, ThatFraction))



tValues = data.frame()
for(deletion_rate_ in unique(dataNormJudg$deletion_rate)) {
  for(predictability_weight_ in unique(dataNormJudg$predictability_weight)) {
	  d2 = dataNormJudg %>% filter(deletion_rate==deletion_rate_, predictability_weight==predictability_weight_)
	  if(nrow(d2) > 1) {
		  if(length(unique(d2$ID)) > 1) {
          tValue = coef(summary(lmer(Surprisal ~ RatioSC + (1|Noun) + (1+RatioSC|ID), data=d2 %>% filter(Region == "V1_0") %>% mutate(ID=as.factor(ID)))))[2,3]
		  } else {
          tValue = coef(summary(lm(Surprisal ~ RatioSC, data=d2 %>% filter(Region == "V1_0") %>% mutate(ID=as.factor(ID)))))[2,3]
		  }
          tValues = rbind(tValues, data.frame(predictability_weight=c(predictability_weight_), deletion_rate=c(deletion_rate_), tValue=c(paste("t =",round(tValue, 4))), yPosition=c(max((d2 %>% filter(Region == "V1_0") %>%  group_by(Noun, RatioSC) %>% summarise(Surprisal=mean(Surprisal)))$Surprisal))))
	  }
}
}

dataV1 = dataNormJudg %>% filter(Region == "V1_0")  %>% group_by(Noun, RatioSC, predictability_weight, deletion_rate) %>% summarise(Surprisal=mean(Surprisal))
ggplot(dataV1, aes(x=RatioSC, y=Surprisal)) + geom_smooth(method="lm") + geom_point()  + geom_label(data=tValues, aes(x=-2, y=yPosition, label=tValue), inherit.aes=FALSE, parse=FALSE)+ facet_grid(deletion_rate~predictability_weight, scales="free") + theme_bw() + xlab("log P(SC|the NOUN)") + ylab("V1 Surprisal")
ggsave("figures/surprisals_NormJudg_OnlyGramm_V1_grid.pdf", height=10, width=6)

write.table(dataV1, file="output/analyze_Judg_VizV1Surp.R.tsv", sep="\t")


ggplot(dataV1, aes(x=RatioSC, y=Surprisal)) + geom_smooth(method="lm") + geom_point(alpha=0.1)  + facet_grid(deletion_rate~predictability_weight, scales="free") + theme_bw() + xlab("log P(SC|the NOUN)") + ylab("V1 Surprisal")
ggsave("figures/surprisals_NormJudg_OnlyGramm_V1_grid_nots.pdf", height=4, width=4)


