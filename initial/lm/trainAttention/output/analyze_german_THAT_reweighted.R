library(dplyr)
library(tidyr)
library(ggplot2)


data = read.csv("results_german_THAT_reweighted.tsv", sep="\t")



plot = ggplot(data=data %>% filter(!is.na(model2)), aes(x=pred_weight, y=del_rate)) + geom_tile(aes(fill=model2))
plot = plot +  scale_fill_gradient2() + facet_wrap(~denoiser)
ggsave(plot, file="model2_german_THAT_reweighted.pdf")


plot = ggplot(data=data %>% filter(!is.na(model1)), aes(x=pred_weight, y=del_rate)) + geom_tile(aes(fill=model1))
plot = plot +  scale_fill_gradient2() + facet_wrap(~denoiser)
ggsave(plot, file="model1_german_THAT_reweighted.pdf")


plot = ggplot(data=data %>% filter(!is.na(sanity2)), aes(x=pred_weight, y=del_rate)) + geom_tile(aes(fill=sanity2))
plot = plot +  scale_fill_gradient2() + facet_wrap(~denoiser)
ggsave(plot, file="sanity2_german_THAT_reweighted.pdf")


plot = ggplot(data=data %>% filter(!is.na(sanity1)), aes(x=pred_weight, y=del_rate)) + geom_tile(aes(fill=sanity1))
plot = plot +  scale_fill_gradient2() + facet_wrap(~denoiser)
ggsave(plot, file="sanity1_german_THAT_reweighted.pdf")



