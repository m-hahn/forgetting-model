data = read.csv("collect.py.tsv", sep=" ", header=F)
names(data) <- c("beta", "EmbeddingRate", "Depth", "Surprisal", "Run")

library(ggplot2)
library(tidyr)
library(dplyr)

library(ggplot2)

data$EmbeddingRate = factor(data$EmbeddingRate, levels=c("Low", "High"))

plot = ggplot(data %>% filter(beta >= 0.1), aes(x=EmbeddingRate, y=Surprisal, color=Depth, group=paste(Run,Depth))) + geom_line() + facet_wrap(~beta) + theme_bw()
ggsave(plot, file="bottleneck.pdf", width=4, height=4)
