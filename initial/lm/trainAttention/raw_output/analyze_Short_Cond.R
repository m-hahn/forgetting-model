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

dataM = data %>% filter(Script == "script__VNStims_3") %>% group_by(Region, ID, Condition, deletion_rate, predictability_weight, Noun) %>% summarise(Surprisal = mean(Surprisal), Ratio=mean(Ratio, na.rm=TRUE), RatioSC=mean(RatioSC, na.rm=TRUE), ThatFraction=mean(ThatFraction, na.rm=TRUE))

plot = ggplot(dataM %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate, scales="free")
plot = ggplot(dataM %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFraction=mean(ThatFraction)), aes(x=Ratio, y=ThatFraction, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate, scales="free")


plot = ggplot(dataM %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + facet_grid(predictability_weight~deletion_rate, scales="free")

plot = ggplot(data %>% filter(Script == "script__VNStims_3", Region == "V1_0") %>% group_by(ID, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFraction=mean(ThatFraction)), aes(x=Ratio, y=ThatFraction, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_wrap(~ID, scales="free")



dataM = data %>% filter(Script == "script__VNStims") %>% group_by(Region, ID, Condition, deletion_rate, predictability_weight, Noun) %>% summarise(Surprisal = mean(Surprisal), Ratio=mean(Ratio, na.rm=TRUE), RatioSC=mean(RatioSC, na.rm=TRUE), ThatFraction=mean(ThatFraction, na.rm=TRUE))

plot = ggplot(dataM %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate, scales="free")
plot = ggplot(dataM %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFraction=mean(ThatFraction)), aes(x=Ratio, y=ThatFraction, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate, scales="free")


dataC = data %>% mutate(Ratio.C = Ratio-mean(Ratio), HasSC = !grepl("NoSC", Condition), HasRC = grepl("SCRC", Condition), HasSC.C = HasSC - mean(HasSC), Compatible = !grepl("ncomp", Condition), predictability_weight.C=predictability_weight-mean(predictability_weight), deletion_rate.C=deletion_rate-mean(deletion_rate))
dataC$HasRC.C = resid(lm(HasRC ~ HasSC, data=dataC))
dataC$Compatible.C = resid(lm(Compatible ~ HasSC, data=dataC))
summary(lmer(Surprisal ~ Compatible.C + HasRC.C * Ratio.C + HasSC.C * Ratio.C + predictability_weight.C + deletion_rate.C + (1|Noun) + (1|ID), data=dataC %>% filter(Script == "script__VNStims")))



dataM = data %>% group_by(Script, Region, ID, Condition, deletion_rate, predictability_weight, Noun) %>% summarise(Surprisal = mean(Surprisal), Ratio=mean(Ratio, na.rm=TRUE), RatioSC=mean(RatioSC, na.rm=TRUE), ThatFraction=mean(ThatFraction, na.rm=TRUE))

plot = ggplot(dataM %>% group_by(deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_wrap(~deletion_rate)
plot = ggplot(dataM %>% group_by(deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth() + geom_text(aes(label=Noun)) + facet_wrap(~deletion_rate)

plot = ggplot(dataM %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth() + geom_text(aes(label=Noun)) + facet_grid(predictability_weight~deletion_rate)



plot = ggplot(dataM %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun)) + facet_grid(predictability_weight~deletion_rate)


plot = ggplot(dataM %>% filter(Region == "V2_1") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_point() + facet_grid(predictability_weight~deletion_rate, scales="free")


plot = ggplot(dataM %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate, scales="free")

# TODO there are weird patterns here
plot = ggplot(dataM %>% filter(Script == "script__VNStims", Region == "V2_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate, scales="free")

plot = ggplot(dataM %>% filter(Script != "script__VNStims", Region == "V2_1") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate, scales="free")
plot = ggplot(dataM %>% filter(Script != "script__VNStims", Region == "V2_1") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFraction=mean(ThatFraction)), aes(x=Ratio, y=ThatFraction, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate, scales="free")


plot = ggplot(dataM %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFraction=mean(ThatFraction)), aes(x=Ratio, y=ThatFraction, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate, scales="free")



corrsComp = dataM %>% filter(Region == "V1_0", !grepl("NoSC", Condition)) %>% group_by(Compatible, HasRC, deletion_rate, predictability_weight, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)) %>% group_by(deletion_rate, predictability_weight) %>% summarise(compt = coef(summary(lm(Surprisal~Compatible+Ratio+HasRC)))[2,3])


corrs = dataM %>% filter(Region == "V1_0", !grepl("NoSC", Condition)) %>% group_by(deletion_rate, predictability_weight, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)) %>% group_by(deletion_rate, predictability_weight) %>% summarise(c = cor(Ratio, Surprisal), p = cor.test(Ratio, Surprisal, alternative="less")$statistic)

plot = ggplot(dataM %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_label(data=corrs, aes(label=paste("t =", round(p, 5)), x=-3, y= 10, group=NULL, color=NULL))
plot = plot + facet_grid(predictability_weight~deletion_rate, scales="free") + theme_bw()


plot = ggplot(dataM %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_label(data=corrsComp, aes(label=paste("t =", round(compt, 5)), x=-3, y= 10, group=NULL, color=NULL))
plot = plot + facet_grid(predictability_weight~deletion_rate, scales="free")


corrsComp = dataM %>% filter(Region == "V1_0", !grepl("NoSC", Condition)) %>% group_by(Compatible, HasRC, deletion_rate, predictability_weight, Noun, Ratio) %>% summarise(ThatFraction=mean(ThatFraction)) %>% group_by(deletion_rate, predictability_weight) %>% summarise(compt = coef(summary(lm(ThatFraction~Compatible+Ratio+HasRC)))[2,3])
plot = ggplot(dataM %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFraction=mean(ThatFraction)), aes(x=Ratio, y=ThatFraction, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_label(data=corrsComp, aes(label=paste("t =", round(compt, 5)), x=-3, y= 0.5, group=NULL, color=NULL))
plot = plot + facet_grid(predictability_weight~deletion_rate, scales="free")



dataM = dataM %>% mutate(HasSC = !grepl("NoSC", Condition), HasRC = grepl("SCRC", Condition)) %>% mutate(Condition2 = ifelse(!HasSC, "NoSC", ifelse(HasRC, "SCRC", "SC")), Compatible = !grepl("ncomp", Condition))

plot = ggplot(dataM %>% filter(Region == "V1_0") %>% group_by(predictability_weight, deletion_rate, Condition2, Noun, Ratio) %>% summarise(Surprisal=mean(Surprisal)), aes(x=Ratio, y=Surprisal, group=Condition2, color=Condition2)) + geom_smooth(method="lm") + geom_label(data=corrs, aes(label=paste("t =", round(p, 5)), x=-3, y= 10, group=NULL, color=NULL))
plot = plot + facet_grid(predictability_weight~deletion_rate, scales="free") + theme_bw()





#dataM %>% filter(Region == "V1_0", !grepl("NoSC", Condition)) %>% group_by(Condition, ID) %>% summarise(c = cor(Ratio, Surprisal)) 
#dataM %>% filter(Region == "V1_0", !grepl("NoSC", Condition)) %>% group_by(Condition, ID) %>% summarise(c = coef(lmer(Surprisal ~ Ratio + (1|ID))) 


