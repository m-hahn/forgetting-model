library(tidyr)
library(dplyr)



data = read.csv("averages_Short_Cond.tsv", quote='"', sep="\t")


counts = unique(read.csv("~/forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", sep="\t"))
counts$Ratio = log(counts$CountThat) - log(counts$CountBare)


scrc = read.csv("/home/user/forgetting/corpus_counts/wikipedia/RC_annotate/results/collectResults.py.tsv", sep="\t")
scrc = scrc %>% mutate(SC_Bias = (SC+1e-10)/(SC+RC+Other+3e-10))
scrc = scrc %>% mutate(Log_SC_Bias = log(SC_Bias))

counts = merge(counts, scrc, by=c("Noun")) %>% mutate(RatioSC = Ratio + Log_SC_Bias)



data = merge(data, counts, by=c("Noun"), all.x=TRUE)


library(ggplot2)


library(lme4)


dataM = data %>% group_by(Region, ID, Condition, deletion_rate, predictability_weight, Noun) %>% summarise(Surprisal = mean(Surprisal), Ratio=mean(Ratio, na.rm=TRUE), RatioSC=mean(RatioSC, na.rm=TRUE), ThatFraction=mean(ThatFraction, na.rm=TRUE))

plot = ggplot(dataM %>% group_by(deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_wrap(~deletion_rate)
plot = ggplot(dataM %>% group_by(deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth() + geom_text(aes(label=Noun)) + facet_wrap(~deletion_rate)

plot = ggplot(dataM %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth() + geom_text(aes(label=Noun)) + facet_grid(predictability_weight~deletion_rate)



plot = ggplot(dataM %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_grid(predictability_weight~deletion_rate)


plot = ggplot(dataM %>% filter(Region == "V2_1") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + facet_grid(predictability_weight~deletion_rate, scales="free")


plot = ggplot(dataM %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + facet_grid(predictability_weight~deletion_rate, scales="free")





