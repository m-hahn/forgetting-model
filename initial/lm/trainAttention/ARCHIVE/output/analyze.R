library(dplyr)
library(tidyr)
library(ggplot2)


data = read.csv("results.tsv", sep="\t")



plot = ggplot(data=data, aes(x=pred_weight, y=del_rate)) + geom_tile(aes(fill=model2))
plot = plot +  scale_fill_gradient2()
ggsave(plot, file="model2.pdf")


plot = ggplot(data=data, aes(x=pred_weight, y=del_rate)) + geom_tile(aes(fill=model1))
plot = plot +  scale_fill_gradient2()
ggsave(plot, file="model1.pdf")


