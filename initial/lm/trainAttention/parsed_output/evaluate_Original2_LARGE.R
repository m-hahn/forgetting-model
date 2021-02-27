data = read.csv("collectCompletionParseStats_Original2_LARGE.py.txt", sep="\t")

counts = read.csv("../../../../../forgetting/fromCorpus_counts.csv", sep="\t")
counts$Ratio = counts$True_False - counts$False_False



data = merge(data, counts, by=c("Noun"), all.x=TRUE)

library(ggplot2)
library(dplyr)
library(tidyr)

plot = ggplot(data %>% group_by(DeletionRate, Noun, Ratio) %>% summarise(complete=mean(complete/(complete+incomplete))), aes(x=Ratio, y=1-complete)) + geom_smooth(method="lm") + geom_point() + facet_wrap(~DeletionRate) + xlab("log P(that|the NOUN)") + ylab("Continuations with Missing Verb") + theme_bw()
ggsave(plot, file="figures/incomplete_by_delRate_LARGE.pdf", width=5, height=5)

plot = ggplot(data %>% group_by(DeletionRate, Noun, Ratio) %>% summarise(That=mean(That/(That+NoThat))), aes(x=Ratio, y=That)) + geom_point() + facet_wrap(~DeletionRate)



data %>% group_by(DeletionRate, Noun, Ratio) %>% summarise(complete=mean(complete/(complete+incomplete))) %>% group_by(DeletionRate) %>% summarise(C=cor(complete, Ratio))

data %>% group_by(PredWeight, Noun, Ratio) %>% summarise(complete=mean(complete/(complete+incomplete))) %>% group_by(PredWeight) %>% summarise(C=cor(complete, Ratio))

plot = ggplot(data %>% group_by(PredWeight, Noun, Ratio) %>% summarise(complete=mean(complete/(complete+incomplete))), aes(x=Ratio, y=1-complete)) + geom_smooth(method="lm") + geom_point() + facet_wrap(~PredWeight) + xlab("log P(that|the NOUN)") + ylab("Continuations with Missing Verb") + theme_bw()
ggsave(plot, file="figures/incomplete_by_predWeight_LARGE.pdf", width=5, height=5)

data %>% group_by(PredWeight, Noun, Ratio) %>% summarise(complete=mean(complete/(complete+incomplete))) %>% group_by(PredWeight) %>% summarise(C=cor.test(complete, Ratio)$p.value)

data %>% group_by(DeletionRate, Noun, Ratio) %>% summarise(complete=mean(complete/(complete+incomplete))) %>% group_by(DeletionRate) %>% summarise(C=cor.test(complete, Ratio)$p.value)

