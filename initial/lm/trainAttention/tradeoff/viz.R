data = read.csv("autoencoderTabulateResultsByRateWeight_OnlyNewModels.py_Erasure.tsv", sep="\t")

library(tidyr)
library(dplyr)
library(ggplot2)

plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(reward = min(reward)) %>% mutate(predictability_weight=as.factor(predictability_weight)), aes(x=deletion_rate, y=reward, group=predictability_weight, color=predictability_weight)) + geom_line() + theme_bw() + xlab("Forgetting Rate (delta)") + ylab("Objective Function")
ggsave(plot, file="delta-objective.pdf", width=5, height=2)

plot = ggplot(data %>% filter(predictability_weight > 0) %>% group_by(deletion_rate, predictability_weight) %>% summarise(predictability = min(predictability)) %>% mutate(predictability_weight=as.factor(predictability_weight)), aes(x=deletion_rate, y=predictability, group=predictability_weight, color=predictability_weight)) + geom_line() + theme_bw() + xlab("Forgetting Rate (delta)") + ylab("Average of -log P(X(t+1)|Y)")
ggsave(plot, file="delta-prediction.pdf", width=5, height=2)

plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(reconstructability = min(reconstructability)) %>% mutate(predictability_weight=as.factor(predictability_weight)), aes(x=deletion_rate, y=reconstructability, group=predictability_weight, color=predictability_weight)) + geom_line() + theme_bw() + xlab("Forgetting Rate (delta)") + ylab("Average of -log P(X|Y)")
ggsave(plot, file="delta-reconstruction.pdf", width=5, height=2)

