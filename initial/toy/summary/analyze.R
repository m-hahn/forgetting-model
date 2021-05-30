data = read.csv("collect.py.tsv", sep=" ", header=F)
names(data) <- c("beta", "EmbeddingRate", "Depth", "Surprisal", "Run")

library(ggplot2)
library(tidyr)
library(dplyr)

library(ggplot2)

plot = ggplot(data, aes(x=EmbeddingRate, y=Surprisal, color=Depth, group=paste(Run,Depth))) + geom_line() + facet_wrap(~beta)

