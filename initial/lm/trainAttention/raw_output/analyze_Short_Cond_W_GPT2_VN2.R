library(tidyr)
library(dplyr)



data = read.csv("averages_Short_Cond_W_GPT2_VN2.tsv", quote='"', sep="\t")


counts = unique(read.csv("~/forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", sep="\t"))
counts$Ratio = log(counts$CountThat) - log(counts$CountBare)


scrc = read.csv("/home/user/forgetting/corpus_counts/wikipedia/RC_annotate/results/collectResults.py.tsv", sep="\t")
scrc = scrc %>% mutate(SC_Bias = (SC+1e-10)/(SC+RC+Other+3e-10))
scrc = scrc %>% mutate(Log_SC_Bias = log(SC_Bias))

counts = merge(counts, scrc, by=c("Noun")) %>% mutate(RatioSC = Ratio + Log_SC_Bias)



data = merge(data, counts, by=c("Noun"), all.x=TRUE)


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

plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/predictions-surprisal_VN.pdf", width=5, height=5)

plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC,  predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)) %>% mutate(ThatFractionReweighted=ifelse(HasSC, ThatFractionReweighted, NA)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Posterior Belief Recovering 'that'") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/predictions-that_VN.pdf", width=5, height=5)


plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/conceptual-predictions-surprisal_VN.pdf", width=2, height=2)


plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC,  Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)) %>% mutate(ThatFractionReweighted=ifelse(HasSC, ThatFractionReweighted, NA)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Posterior Belief Recovering 'that'") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/conceptual-predictions-that_VN.pdf", width=5, height=5)




plot = ggplot(unique(data %>% filter(Region == "V1_0") %>% select(predictability_weight, deletion_rate, ID)) %>% group_by(predictability_weight, deletion_rate) %>% summarise(Models=NROW(ID)), aes(x=deletion_rate, y=predictability_weight, fill=Models)) + geom_tile() + scale_fill_gradient2()
ggsave(plot, file="figures/model-runs", width=5, height=5)







plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(ID, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm", se=FALSE) + facet_wrap(~predictability_weight+deletion_rate+ID)


plot = ggplot(data %>% filter(Region == "V1_0", grepl("GPT2L", Script)) %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm", se=FALSE) + facet_grid(predictability_weight~deletion_rate+Script)
ggsave(plot, file="figures/Q_L_surp.pdf", width=10, height=10)
plot = ggplot(data %>% filter(Region == "V1_0", grepl("GPT2L", Script)) %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script)
ggsave(plot, file="figures/Q_L_that.pdf", width=10, height=10)



plot = ggplot(data %>% filter(Region == "V1_0", grepl("GPT2M", Script)) %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm", se=FALSE) + facet_grid(predictability_weight~deletion_rate+Script)
ggsave(plot, file="figures/Q_M_surp.pdf", width=10, height=10)

plot = ggplot(data %>% filter(Region == "V1_0", grepl("GPT2M", Script)) %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script)
ggsave(plot, file="figures/Q_M_that.pdf", width=10, height=10)


plot = ggplot(data %>% filter(Region == "V1_0", Script == "script__Q_3_W_GPT2") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm", se=FALSE) + facet_grid(predictability_weight~deletion_rate+Script)
ggsave(plot, file="figures/Q_S_surp.pdf", width=10, height=10)

plot = ggplot(data %>% filter(Region == "V1_0", Script == "script__Q_3_W_GPT2") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script)
ggsave(plot, file="figures/Q_S_that.pdf", width=10, height=10)



plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_grid(predictability_weight~deletion_rate+Script)
plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script)

plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm", se=FALSE) + facet_grid(predictability_weight~deletion_rate)
plot = ggplot(data %>% filter(Region == "V1_0", grepl("GPT2L", Script)) %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm", se=FALSE) + facet_grid(predictability_weight~deletion_rate)
plot = ggplot(data %>% filter(Region == "V1_0", grepl("GPT2M", Script)) %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm", se=FALSE) + facet_grid(predictability_weight~deletion_rate)



plot = ggplot(data %>% filter(grepl("VNStims", Script), Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V1_VN_GPT2_Joint.pdf", height=10, width=10)
plot = ggplot(data %>% filter(!grepl("VNStims", Script), Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V1_GPT2_Joint.pdf", height=10, width=10)

plot = ggplot(data %>% filter(grepl("VNStims", Script), Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_That_V1_VN_GPT2_Joint.pdf", height=10, width=10)
plot = ggplot(data %>% filter(!grepl("VNStims", Script), Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_That_V1_GPT2_Joint.pdf", height=10, width=10)






plot = ggplot(data %>% filter(grepl("VNStims", Script), Region == "V1_0", Condition %in% c("SC_compatible", "SC_incompatible")) %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V1_VN_GPT2_SC_Joint.pdf", height=10, width=10)
plot = ggplot(data %>% filter(!grepl("VNStims", Script), Region == "V1_0", Condition %in% c("SC_compatible", "SC_incompatible")) %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V1_GPT2_SC_Joint.pdf", height=10, width=10)

plot = ggplot(data %>% filter(grepl("VNStims", Script), Region == "V1_0", Condition %in% c("SC_compatible", "SC_incompatible")) %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_That_V1_VN_GPT2_SC_Joint.pdf", height=10, width=10)
plot = ggplot(data %>% filter(!grepl("VNStims", Script), Region == "V1_0", Condition %in% c("SC_compatible", "SC_incompatible")) %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_That_V1_GPT2_SC_Joint.pdf", height=10, width=10)




plot = ggplot(data %>% filter(Script == "script__VNStims_3_W_GPT2", Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V1_VN_GPT2.pdf", height=10, width=10)
plot = ggplot(data %>% filter(Script == "script__VNStims_3_W_GPT2", Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script) + theme_bw()
ggsave(plot, file="figures/Compat_That_V1_VN_GPT2.pdf", height=10, width=10)

plot = ggplot(data %>% filter(Script == "script__W_GPT2", Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V1_GPT2.pdf", height=10, width=10)
plot = ggplot(data %>% filter(Script == "script__W_GPT2", Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script) + theme_bw()
ggsave(plot, file="figures/Compat_That_V1_GPT2.pdf", height=10, width=10)

plot = ggplot(data %>% filter(Script == "script__W_GPT2M", Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V1_GPT2M.pdf", height=10, width=10)
plot = ggplot(data %>% filter(Script == "script__W_GPT2M", Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script) + theme_bw()
ggsave(plot, file="figures/Compat_That_V1_GPT2M.pdf", height=10, width=10)

plot = ggplot(data %>% filter(Script == "script__W_GPT2L", Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V1_GPT2L.pdf", height=10, width=10)
plot = ggplot(data %>% filter(Script == "script__W_GPT2L", Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script) + theme_bw()
ggsave(plot, file="figures/Compat_That_V1_GPT2L.pdf", height=10, width=10)

plot = ggplot(data %>% filter(Script == "script__VNStims_3_W_GPT2M", Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V1_VN_GPT2M.pdf", height=10, width=10)
plot = ggplot(data %>% filter(Script == "script__VNStims_3_W_GPT2M", Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script) + theme_bw()
ggsave(plot, file="figures/Compat_That_V1_VN_GPT2M.pdf", height=10, width=10)
plot = ggplot(data %>% filter(Script == "script__VNStims_3_W_GPT2L", Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V1_VN_GPT2L.pdf", height=10, width=10)
plot = ggplot(data %>% filter(Script == "script__VNStims_3_W_GPT2L", Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script) + theme_bw()
ggsave(plot, file="figures/Compat_That_V1_VN_GPT2L.pdf", height=10, width=10)






plot = ggplot(data %>% filter(grepl("VNStims", Script), Region == "V2_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V2_0_VN_GPT2_Joint.pdf", height=10, width=10)
plot = ggplot(data %>% filter(!grepl("VNStims", Script), Region == "V2_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V2_0_GPT2_Joint.pdf", height=10, width=10)

plot = ggplot(data %>% filter(grepl("VNStims", Script), Region == "V2_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_That_V2_0_VN_GPT2_Joint.pdf", height=10, width=10)
plot = ggplot(data %>% filter(!grepl("VNStims", Script), Region == "V2_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_That_V2_0_GPT2_Joint.pdf", height=10, width=10)




plot = ggplot(data %>% filter(grepl("VNStims", Script), Region == "V2_1") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V2_1_VN_GPT2_Joint.pdf", height=10, width=10)
plot = ggplot(data %>% filter(!grepl("VNStims", Script), Region == "V2_1") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V2_1_GPT2_Joint.pdf", height=10, width=10)

plot = ggplot(data %>% filter(grepl("VNStims", Script), Region == "V2_1") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_That_V2_1_VN_GPT2_Joint.pdf", height=10, width=10)
plot = ggplot(data %>% filter(!grepl("VNStims", Script), Region == "V2_1") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_That_V2_1_GPT2_Joint.pdf", height=10, width=10)



plot = ggplot(data %>% filter(grepl("VNStims", Script), Region == "V2_2") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V2_2_VN_GPT2_Joint.pdf", height=10, width=10)
plot = ggplot(data %>% filter(!grepl("VNStims", Script), Region == "V2_2") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_Surp_V2_2_GPT2_Joint.pdf", height=10, width=10)

plot = ggplot(data %>% filter(grepl("VNStims", Script), Region == "V2_2") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_That_V2_2_VN_GPT2_Joint.pdf", height=10, width=10)
plot = ggplot(data %>% filter(!grepl("VNStims", Script), Region == "V2_2") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=paste(Condition, Script), linetype=Script, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate) + theme_bw()
ggsave(plot, file="figures/Compat_That_V2_2_GPT2_Joint.pdf", height=10, width=10)









plot = ggplot(data %>% filter(Script == "script__VNStims_3_W_GPT2", Region == "V1_0", Condition %in% c("SC_compatible", "SC_incompatible")) %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script)





plot = ggplot(data %>% filter(Script == "script__VNStims_3_W_GPT2", Region == "V2_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script)
plot = ggplot(data %>% filter(Script == "script__VNStims_3_W_GPT2", Region == "V2_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script)
plot = ggplot(data %>% filter(Script == "script__W_GPT2", Region == "V2_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script)
plot = ggplot(data %>% filter(Script == "script__W_GPT2", Region == "V2_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script)
plot = ggplot(data %>% filter(Script == "script__VNStims_3_W_GPT2M", Region == "V2_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script)
plot = ggplot(data %>% filter(Script == "script__VNStims_3_W_GPT2M", Region == "V2_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate+Script)







plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_grid(predictability_weight~deletion_rate)

plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_grid(predictability_weight~deletion_rate)

plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFraction=mean(ThatFraction)), aes(x=Ratio, y=ThatFraction, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_grid(predictability_weight~deletion_rate)


plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(Script, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_grid(predictability_weight+Script ~deletion_rate)

library(brms)
#summary(brm(SurprisalReweighted ~ Ratio + Condition + (1+Condition|Noun) + (1+Ratio+Condition|ID), data=data))
#                         Estimate Est.Error l-95% CI u-95% CI Rhat Bulk_ESS
#Intercept                    8.74      0.10     8.53     8.93 1.00     1252
#Ratio                       -0.17      0.03    -0.22    -0.11 1.00     1580
#ConditionSC_incompatible    -0.12      0.03    -0.18    -0.06 1.00     3077


