library(tidyr)
library(dplyr)



data = read.csv("averages_Short_Cond_W_GPT2_QC_Uniform.tsv", quote='"', sep="\t")


counts = unique(read.csv("~/forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", sep="\t"))
counts$Ratio = log(counts$CountThat) - log(counts$CountBare)


scrc = read.csv("/home/user/forgetting/corpus_counts/wikipedia/RC_annotate/results/collectResults.py.tsv", sep="\t")
scrc = scrc %>% mutate(SC_Bias = (SC+1e-10)/(SC+RC+Other+3e-10))
scrc = scrc %>% mutate(Log_SC_Bias = log(SC_Bias))

counts = merge(counts, scrc, by=c("Noun")) %>% mutate(RatioSC = Ratio + Log_SC_Bias)



data = merge(data, counts, by=c("Noun"), all.x=TRUE)

data = data %>% mutate(SurprisalReweighted=SurprisalReweighted/log(2))

library(ggplot2)


library(lme4)


plot = ggplot(data %>% filter(Region == "V2_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_grid(predictability_weight~deletion_rate)

plot = ggplot(data %>% filter(Region == "V2_1") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_grid(predictability_weight~deletion_rate)

# By Script (and thus stimuli)
plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_grid(predictability_weight~deletion_rate+Script)
plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm", se=FALSE) + facet_grid(predictability_weight~deletion_rate+Script)


plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate)
plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate)



data$compatible = grepl("_compatible", data$Condition)
data$HasSC = !grepl("NoSC", data$Condition)
data$HasRC = grepl("RC", data$Condition)

data$HasSCHasRC = (paste(data$HasSC, data$HasRC, sep="_"))

plot = ggplot(data %>% filter(Region == "V1_0") %>% mutate(deletion_rate=20*(1-deletion_rate)) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Surprisal (bits)") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/predictions-surprisal-uniform_Bits.pdf", width=7, height=1.5)

plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC,  deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)) %>% mutate(ThatFractionReweighted=ifelse(HasSC, ThatFractionReweighted, NA)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Posterior Belief Recovering 'that'") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/predictions-that-uniform_Bits.pdf", width=5, height=5)


