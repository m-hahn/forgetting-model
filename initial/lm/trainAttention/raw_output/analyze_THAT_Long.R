

data = read.csv("averages_Long.tsv", quote='"', sep="\t")



library(ggplot2)

counts = read.csv("../../../../../forgetting/fromCorpus_counts.csv", sep="\t")
counts$Ratio = counts$True_False - counts$False_False



data = merge(data, counts, by=c("Noun"), all.x=TRUE)

library(lme4)

data$Condition.C = ifelse(data$Condition == "u", -0.5, 0.5)

library(tidyr)
library(dplyr)

data = data %>% filter(Region == "V2", Condition == "g")

summary(lmer(ThatFraction ~ Condition.C*Ratio + (1+Condition.C|Noun), data=data))

data$deletion_rate_coarse = round(data$deletion_rate*10)/10

correlations_V2_ByID = data %>% group_by(ID, predictability_weight, deletion_rate_coarse) %>% summarise(Correlation = cor(ThatFraction, Ratio))


plot = ggplot(data=correlations_V2_ByID, aes(x=predictability_weight, y=deletion_rate_coarse)) + geom_tile(aes(fill=Correlation))
plot = plot +  scale_fill_gradient2() #+ facet_wrap(~denoiser)


