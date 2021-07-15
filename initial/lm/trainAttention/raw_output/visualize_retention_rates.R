data = read.csv("collect12_NormJudg_Short_Cond_W_GPT2_QC_Attention.py.tsv", sep="\t")

library(ggplot2)
library(dplyr)
library(tidyr)


data = data %>% filter(Script %in% c("script__QC_3_W_GPT2M", "script__QC_3_W_GPT2L"))
data = data %>% filter(Word %in% c("the", "that", "of", "by", "fact"))

data = data %>% filter(!(Distance %in% c(0, 20)))

plot = ggplot(data %>% group_by(Word, Distance, deletion_rate, predictability_weight) %>% summarise(AvailableRate=mean(AvailableRate)), aes(x=-20+Distance, y=AvailableRate, group=Word, color=Word)) + geom_line() + facet_grid(predictability_weight~deletion_rate)


plot = ggplot(data %>% group_by(Word, Distance, predictability_weight) %>% summarise(AvailableRate=mean(AvailableRate)), aes(x=-20+Distance, y=AvailableRate, group=Word, color=Word)) + geom_line() + facet_grid(~predictability_weight) + theme_bw()



plot = ggplot(data %>% group_by(Word, Distance, deletion_rate) %>% summarise(AvailableRate=mean(AvailableRate)), aes(x=-20+Distance, y=AvailableRate, group=Word, color=Word)) + geom_line() + facet_grid(~deletion_rate) + theme_bw()


plot = ggplot(data, aes(x=-20+Distance, y=AvailableRate, group=Word, color=Word)) + geom_smooth(se=F) + facet_grid(~deletion_rate) + theme_bw()



plot = ggplot(data %>% group_by(Word, Distance) %>% summarise(AvailableRate=mean(AvailableRate)), aes(x=-20+Distance, y=AvailableRate, group=Word, color=Word)) + geom_line() +  theme_bw() + xlab("Distance in Past") + ylab("Retention Rate")
ggsave(plot, file="figures/retentionRates.pdf", height=1.5, width=3)



plot = ggplot(data %>% group_by(Word, Distance) %>% summarise(AvailableRate=mean(AvailableRate)), aes(x=-20+Distance, y=AvailableRate, group=Word, color=Word)) + geom_line() +  theme_bw() + xlab("Distance in Past") + ylab("Retention Rate") + geom_text(data= data  %>% group_by(Word, Distance) %>% summarise(AvailableRate=mean(AvailableRate)) %>% filter(-20+Distance == -10), aes(x=-20+Distance, y=AvailableRate+0.1, label=Word))+ theme (legend.position = "none")
ggsave(plot, file="figures/retentionRates_noLegend.pdf", height=1.5, width=3)




data = data %>% mutate(deletion_rate_rough = (deletion_rate > 0.5))
data = data %>% mutate(predictability_weight_rough = (predictability_weight > 0.5))

plot = ggplot(data %>% group_by(Word, Distance, deletion_rate_rough, predictability_weight_rough) %>% summarise(AvailableRate=mean(AvailableRate)), aes(x=-20+Distance, y=AvailableRate, group=Word, color=Word)) + geom_line() + facet_grid(predictability_weight_rough~deletion_rate_rough) + theme_bw()




plot = ggplot(data, aes(x=-20+Distance, y=AvailableRate, group=Word, color=Word)) + geom_smooth(se=F) + facet_grid(predictability_weight_rough~deletion_rate_rough) + theme_bw()


