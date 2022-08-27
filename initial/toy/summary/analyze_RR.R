data = read.csv("collect_RR.py.tsv", sep=" ", header=F)
names(data) <- c("delta", "EmbeddingRate", "Depth", "Surprisal")

library(ggplot2)
library(tidyr)
library(dplyr)

library(ggplot2)

data$EmbeddingRate = factor(data$EmbeddingRate, levels=c("Low", "High"))


plot = ggplot(data %>% group_by(delta, EmbeddingRate, Depth) %>% summarise(Surprisal=mean(Surprisal)) %>% group_by() %>% mutate(delta=8*(1-delta)), aes(x=EmbeddingRate, y=1.5+Surprisal, color=Depth, group=paste(Depth))) + geom_line() + facet_wrap(~delta) + theme_bw() + ylab("Surprisal")
ggsave(plot, file="resource-rational_Integer.pdf", width=4, height=4)

plot = ggplot(data %>% group_by(delta, EmbeddingRate, Depth) %>% summarise(Surprisal=mean(Surprisal)), aes(x=EmbeddingRate, y=1.5+Surprisal, color=Depth, group=paste(Depth))) + geom_line() + facet_wrap(~delta) + theme_bw() + ylab("Surprisal")
ggsave(plot, file="resource-rational.pdf", width=4, height=4)



