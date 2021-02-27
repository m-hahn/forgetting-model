#data = read.csv("semiautomatically-results.tsv", sep="\t")
#summary(data)
#data$CompleteRatio = data$Complete/(data$Complete+data$Incomplete+data$Unknown)
library(tidyr)
library(dplyr)
#data %>% filter(Noun == "fact") %>% summarise(CompleteRatio = mean(CompleteRatio))
#data %>% filter(Noun == "report") %>% summarise(CompleteRatio = mean(CompleteRatio))
#data %>% filter(Noun == "story") %>% summarise(CompleteRatio = mean(CompleteRatio))
#data %>% filter(Noun == "admission") %>% summarise(CompleteRatio = mean(CompleteRatio))
#data %>% filter(Noun == "belief") %>% summarise(CompleteRatio = mean(CompleteRatio))
#data %>% filter(Noun == "complaint") %>% summarise(CompleteRatio = mean(CompleteRatio))
#data %>% filter(Noun == "assumption") %>% summarise(CompleteRatio = mean(CompleteRatio))
#data %>% group_by(Noun == "assumption") %>% summarise(CompleteRatio = mean(CompleteRatio))
#u = data %>% group_by(Noun) %>% summarise(CompleteRatio = mean(CompleteRatio))
#u[order(u$CompleteRatio),]
#print(u[order(u$CompleteRatio),], n=50)
                                                                                                          
counts = read.csv("../../../../../forgetting/fromCorpus_counts.csv", sep="\t")
                                                                                               
counts$Ratio = counts$True_False - counts$False_False                                  
                                                                                                                                    
#data = merge(u, counts, by=c("Noun"), all.x=TRUE)
#cor(data$Ratio, data$CompleteRatio)
#cor.test(data$Ratio, data$CompleteRatio)
#cor.test(data$Ratio, log(data$CompleteRatio))
library(lme4)
data = read.csv("semiautomatically-normjudg-results.tsv", sep="\t")
data$CompleteRatio = data$Complete/(data$Complete+data$Incomplete+data$Unknown)
data = merge(data, counts, by=c("Noun"), all.x=TRUE)
summary(lmer(CompleteRatio ~ Ratio + (1|ID), data=data))
summary(lmer(CompleteRatio ~ Ratio + (1+Ratio|ID), data=data))
summary(lmer(CompleteRatio ~ Ratio + DeletionRate + (1+Ratio|ID), data=data))
summary(lmer(CompleteRatio ~ Ratio + DeletionRate + PredictabilityWeight + (1+Ratio|ID), data=data))
summary(data$Unknown/(data$Complete+data$Incomplete+data$Unknown))
data$UnknownRatio = (data$Unknown/(data$Complete+data$Incomplete+data$Unknown))
data %>% filter(UnknownRatio > 0.6)
data$CompleteRatio = data$Complete/(data$Complete+data$Incomplete)
summary(lmer(CompleteRatio ~ Ratio + DeletionRate + PredictabilityWeight + (1+Ratio|ID) + (1|Noun), data=data))
#savehistory("semiautomatic.R")

library(ggplot2)

plot = ggplot(data %>% group_by(Ratio, DeletionRate) %>% summarise(CompleteRatio = mean(CompleteRatio, na.rm=TRUE)), aes(x=Ratio, y=CompleteRatio)) + geom_point() + geom_smooth(method="lm") + facet_wrap(~DeletionRate)


plot = ggplot(data %>% group_by(Ratio, PredictabilityWeight) %>% summarise(CompleteRatio = mean(CompleteRatio, na.rm=TRUE)), aes(x=Ratio, y=CompleteRatio)) + geom_point() + geom_smooth(method="lm") + facet_wrap(~PredictabilityWeight)

tValues = data.frame(DeletionRate = c(), PredictabilityWeight = c(), t=c())
for(d in unique(data$DeletionRate)) {
	for(p in unique(data$PredictabilityWeight)) {
		d2 = data %>% filter(DeletionRate == d, PredictabilityWeight == p)
		if(nrow(d2) > 0) {
			if(length(unique(d2$ID)) == 1) {
			   tv = (summary(lm(CompleteRatio ~ Ratio, data=d2)))$coefficients[2,3]
			} else {
   			  tv = (summary(lmer(CompleteRatio ~ Ratio + (1+Ratio|ID) + (1|Noun), data=d2)))$coefficients[2,3]
			}
			tValues = rbind(tValues, data.frame(DeletionRate=c(d), PredictabilityWeight=c(p), t=c(tv)))
		}
	}
}


plot = ggplot(data %>% group_by(Ratio, DeletionRate, PredictabilityWeight) %>% summarise(CompleteRatio = mean(CompleteRatio, na.rm=TRUE)), aes(x=Ratio, y=1-CompleteRatio)) + geom_point(alpha=0.1) + geom_smooth(method="lm") + geom_label(data=tValues, aes(label=paste("t=",round(t,2)), x=(-3.75), y=c(0.8)), size=2) + facet_grid(rows=DeletionRate ~ PredictabilityWeight) + theme_bw() +  theme ( panel.grid.major = element_blank (), panel.grid.minor = element_blank ()) + xlab("log P(SC|the NOUN)") + ylab("Posterior Samples Favoring Ungrammatical Continuation")

ggsave(plot, file="ts-table-normjudg.pdf", height=4, width=3.0)

plot = ggplot(data %>% group_by(Ratio, DeletionRate, PredictabilityWeight) %>% summarise(CompleteRatio = mean(CompleteRatio, na.rm=TRUE)), aes(x=Ratio, y=1-CompleteRatio)) + geom_point(alpha=0.1) + geom_smooth(method="lm") + facet_grid(rows=DeletionRate ~ PredictabilityWeight) + theme_bw() +  theme ( panel.grid.major = element_blank (), panel.grid.minor = element_blank ()) + xlab("log P(SC|the NOUN)") + ylab("Posterior Samples Favoring Ungrammatical Continuation")

ggsave(plot, file="table-normjudg.pdf", height=4, width=3.0)



