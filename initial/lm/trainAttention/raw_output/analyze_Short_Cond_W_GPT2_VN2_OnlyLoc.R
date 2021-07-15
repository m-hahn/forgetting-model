library(tidyr)
library(dplyr)



data = read.csv("averages_Short_Cond_VN2_OnlyLoc.tsv", quote='"', sep="\t")



counts = unique(read.csv("~/forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", sep="\t"))
counts$Ratio = log(counts$CountThat) - log(counts$CountBare)


scrc = read.csv("/home/user/forgetting/corpus_counts/wikipedia/RC_annotate/results/collectResults.py.tsv", sep="\t")
scrc = scrc %>% mutate(SC_Bias = (SC+1e-10)/(SC+RC+Other+3e-10))
scrc = scrc %>% mutate(Log_SC_Bias = log(SC_Bias))

counts = merge(counts, scrc, by=c("Noun")) %>% mutate(RatioSC = Ratio + Log_SC_Bias)



data = merge(data, counts, by=c("Noun"), all.x=TRUE)


library(ggplot2)


library(lme4)


data$compatible = grepl("_compatible", data$Condition)
data$HasSC = !grepl("NoSC", data$Condition)
data$HasRC = grepl("RC", data$Condition)

data$HasSCHasRC = (paste(data$HasSC, data$HasRC, sep="_"))

library(stringr)
#data$Script = str_replace(data$Script, "script__VN2Stims_3_W_", "")
#data$Script = str_replace(data$Script, "_OnlyLoc", "")

#data$Script = as.factor(data$Script)
#data$Script = factor(data$Script, levels=c("GPT2M", "GPT2L", "GPT2XL", "TXL"))

plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(deletion_rate, compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
#ggsave(plot, file="figures/predictions-surprisal-zero_VN.pdf", width=5, height=2)

#
#plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(Script, compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + geom_point() + facet_grid(~Script) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
#                                "TRUE_FALSE"="#00BA38",
#                                "TRUE_TRUE"="#619CFF")) 
#ggsave(plot, file="figures/predictions-surprisal-zero_VN_scatter.pdf", width=5, height=2)
#
#for(model in levels(data$Script)) {
#    plot = ggplot(data %>% filter(Script == model, Region == "V1_0") %>% group_by(Script, compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
#                                    "TRUE_FALSE"="#00BA38",
#                                    "TRUE_TRUE"="#619CFF")) 
#    ggsave(plot, file=paste("figures/predictions-surprisal-zero_VN_", model, ".pdf", sep=""), width=2, height=2)
#   
#    plot = ggplot(data %>% filter(Script == model, Region == "V1_0") %>% group_by(Script, compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + geom_point() + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
#                                    "TRUE_FALSE"="#00BA38",
#                                    "TRUE_TRUE"="#619CFF")) 
#    ggsave(plot, file=paste("figures/predictions-surprisal-zero_VN_", model, "_scatter.pdf", sep=""), width=2, height=2)
#}
#
#


