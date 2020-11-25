data = read.csv("collectCompletionParseStats_Original2.py.txt", sep="\t")

counts = read.csv("../../../../../forgetting/fromCorpus_counts.csv", sep="\t")
counts$Ratio = counts$True_False - counts$False_False



data = merge(data, counts, by=c("Noun"), all.x=TRUE)

library(ggplot2)
library(dplyr)
library(tidyr)

plot = ggplot(data %>% group_by(DeletionRate, Noun, Ratio) %>% summarise(Subj=mean(Subj/(Subj+NotSubj))), aes(x=Ratio, y=1-Subj)) + geom_smooth(method="lm") + geom_point() + facet_wrap(~DeletionRate) + xlab("log P(that|the NOUN)") + ylab("Continuations with Missing Verb") + theme_bw()
ggsave(plot, file="figures/incomplete_by_delRate.pdf", width=5, height=5)

plot = ggplot(data %>% group_by(DeletionRate, Noun, Ratio) %>% summarise(That=mean(That/(That+NoThat))), aes(x=Ratio, y=That)) + geom_point() + facet_wrap(~DeletionRate)



data %>% group_by(DeletionRate, Noun, Ratio) %>% summarise(Subj=mean(Subj/(Subj+NotSubj))) %>% group_by(DeletionRate) %>% summarise(C=cor(Subj, Ratio))

data %>% group_by(PredWeight, Noun, Ratio) %>% summarise(Subj=mean(Subj/(Subj+NotSubj))) %>% group_by(PredWeight) %>% summarise(C=cor(Subj, Ratio))

plot = ggplot(data %>% group_by(PredWeight, Noun, Ratio) %>% summarise(Subj=mean(Subj/(Subj+NotSubj))), aes(x=Ratio, y=1-Subj)) + geom_smooth(method="lm") + geom_point() + facet_wrap(~PredWeight) + xlab("log P(that|the NOUN)") + ylab("Continuations with Missing Verb") + theme_bw()
ggsave(plot, file="figures/incomplete_by_predWeight.pdf", width=5, height=5)

data %>% group_by(PredWeight, Noun, Ratio) %>% summarise(Subj=mean(Subj/(Subj+NotSubj))) %>% group_by(PredWeight) %>% summarise(C=cor.test(Subj, Ratio)$p.value)

data %>% group_by(DeletionRate, Noun, Ratio) %>% summarise(Subj=mean(Subj/(Subj+NotSubj))) %>% group_by(DeletionRate) %>% summarise(C=cor.test(Subj, Ratio)$p.value)

