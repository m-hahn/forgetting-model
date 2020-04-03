data = read.csv("rewards.tsv", sep="\t")


library(tidyr)
library(dplyr)
library(ggplot2)




ggplot(data, aes(x=counter, y=reward)) + geom_line() + facet_wrap(~version+filenum)


ggplot(data %>% filter(rate==0.2), aes(x=counter, y=reward)) + geom_smooth() + facet_wrap(~version+filenum)




