data = read.csv("collectCompletionParseStats_Original.py.txt", sep="\t")

counts = read.csv("../../../../../forgetting/fromCorpus_counts.csv", sep="\t")
counts$Ratio = counts$True_False - counts$False_False



data = merge(data, counts, by=c("Noun"), all.x=TRUE)

library(ggplot2)
library(dplyr)
library(tidyr)

plot = ggplot(data %>% group_by(DeletionRate, Noun, Ratio) %>% summarise(Subj=mean(Subj/(Subj+NotSubj))), aes(x=Ratio, y=Subj)) + geom_smooth() + geom_point() + facet_wrap(~DeletionRate)

plot = ggplot(data %>% group_by(DeletionRate, Noun, Ratio) %>% summarise(That=mean(That/(That+NoThat))), aes(x=Ratio, y=That)) + geom_point() + facet_wrap(~DeletionRate)


