library(tidyr)
library(dplyr)



data = read.csv("prepareMeansByExperiment_ByStimuli.R.tsv", quote='"', sep="\t") %>% filter(StimulusSet == "VAdv")


counts = unique(read.csv("~/forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", sep="\t"))
counts$Ratio = log(counts$CountThat) - log(counts$CountBare)


scrc = read.csv("/home/user/forgetting/corpus_counts/wikipedia/RC_annotate/results/collectResults.py.tsv", sep="\t")
scrc = scrc %>% mutate(SC_Bias = (SC+1e-10)/(SC+RC+Other+3e-10))
scrc = scrc %>% mutate(Log_SC_Bias = log(SC_Bias))

counts = merge(counts, scrc, by=c("Noun")) %>% mutate(RatioSC = Ratio + Log_SC_Bias)



data = merge(data, counts, by=c("Noun"), all.x=TRUE)


library(ggplot2)


library(lme4)



data$compatible = grepl("_co", data$Condition)
data$HasSC = !grepl("NoSC", data$Condition)
data$HasRC = grepl("RC", data$Condition)

data$HasSCHasRC = (paste(data$HasSC, data$HasRC, sep="_"))


plot = ggplot(data %>% filter(predictability_weight==1.0) %>% filter(Experiment=="Experiment2", Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/predictions-surprisal_VN_Experiment2.pdf", width=5, height=5)

plot = ggplot(data %>% filter(Experiment=="Experiment2", Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 




plot = ggplot(data %>% filter(Experiment=="Experiment2", Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/predictions-surprisal_VN_Experiment2.pdf", width=5, height=5)

plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC,  predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)) %>% mutate(ThatFractionReweighted=ifelse(HasSC, ThatFractionReweighted, NA)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Posterior Belief Recovering 'that'") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/predictions-that_VN.pdf", width=5, height=5)


plot = ggplot(data %>% filter(Experiment=="Experiment2", Region == "V1_0", deletion_rate<0.66, deletion_rate>=0.33, predictability_weight>=0.5) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/conceptual-predictions-surprisal_VN.pdf", width=2, height=2)


regions = data.frame(delta_lower = c(0, 0.33, 0.66, 0, 0.33, 0.66),  delta_upper = c(0.33, 0.66, 1, 0.33, 0.66, 1), lambda_lower = c(0, 0, 0, 0.5, 0.5, 0.5), lambda_upper = c(0.5, 0.5, 0.5, 1, 1, 1))


for(experiment in c("Experiment1", "Experiment2")) {
  for(i in (1:nrow(regions))) {
     plot = ggplot(data %>% filter(Experiment==experiment, Region == "V1_0", deletion_rate<regions$delta_upper[[i]], deletion_rate>=regions$delta_lower[[i]], predictability_weight>=regions$lambda_lower[[i]], predictability_weight<= regions$lambda_upper[[i]]) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                     "TRUE_FALSE"="#00BA38",
                                     "TRUE_TRUE"="#619CFF")) 
     ggsave(plot, file=paste("figures/byParamRegion/conceptual-predictions-surprisal_VN_", experiment, "_", i, ".pdf", sep=""), width=2, height=2)
  }

     plot = ggplot(data %>% filter(Experiment==experiment, Region == "V1_0", deletion_rate<0.66, deletion_rate>0.33) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                     "TRUE_FALSE"="#00BA38",
                                     "TRUE_TRUE"="#619CFF")) 
     ggsave(plot, file=paste("figures/byParamRegion/conceptual-predictions-surprisal_VN_", experiment, "_", i, ".pdf", sep=""), width=2, height=2)

}



for(experiment in c("Experiment1", "Experiment2")) {
  plot = ggplot(data %>% filter(Experiment==experiment, Region == "V1_0", deletion_rate>=0.3, deletion_rate <=0.7) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
     ggsave(plot, file=paste("figures/byParamRegion/conceptual-predictions-surprisal_VN_", experiment, "_AcrossLambda.pdf", sep=""), width=5, height=1.7)
}


for(experiment in c("Experiment1", "Experiment2")) {
  plot = ggplot(data %>% mutate(predictability_weight_coarse=predictability_weight>0.5) %>% filter(Experiment==experiment, Region == "V1_0", deletion_rate>=0.3, deletion_rate <=0.7) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight_coarse, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight_coarse~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
     ggsave(plot, file=paste("figures/byParamRegion/conceptual-predictions-surprisal_VN_", experiment, "_ByLambdaRegion.pdf", sep=""), width=5, height=1.7)
}



for(experiment in c("Experiment1", "Experiment2")) {
   dataSmoothed = data.frame()
   params = unique(data %>% select(deletion_rate, predictability_weight))
   for(i in (1:nrow(params))) {
   	del = params$deletion_rate[[i]]
   	pred = params$predictability_weight[[i]]
   	data_ = data %>% filter(Region == "V1_0", Experiment == experiment, abs(deletion_rate-del) <= 0.05, abs(predictability_weight-pred) <= 0.25)
   	data_ = data_ %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))
   	data_$deletion_rate=del
   	data_$predictability_weight=pred
   	dataSmoothed = rbind(dataSmoothed, as.data.frame(data_))
   
   }
   
     plot = ggplot(dataSmoothed %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                   "TRUE_FALSE"="#00BA38",
                                   "TRUE_TRUE"="#619CFF")) 
   
   ggsave(plot, file=paste("figures/byParamRegion/conceptual-predictions-surprisal_VN_", experiment, "_Smoothed.pdf", sep=""), width=10, height=6)

   plot = ggplot(dataSmoothed %>% filter(deletion_rate>0.33, deletion_rate<0.66) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                   "TRUE_FALSE"="#00BA38",
                                   "TRUE_TRUE"="#619CFF")) 
   
   ggsave(plot, file=paste("figures/byParamRegion/conceptual-predictions-surprisal_VN_", experiment, "_Smoothed_MediumDeletionRate.pdf", sep=""), width=6, height=6)
}


# Per Model
ggplot(data %>% filter(Experiment=="Experiment2", predictability_weight==1, Region=="V1_0") %>% mutate(model=paste(predictability_weight, deletion_rate, Script, ID)) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio, model) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_wrap(~model) + theme_bw() + theme(legend.position = "none")



# Per Model
ggplot(data %>% filter(deletion_rate >= 0.4, deletion_rate<=0.55, Experiment=="Experiment2", predictability_weight==0.5, Region=="V1_0") %>% mutate(model=paste(predictability_weight, deletion_rate, Script, ID)) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio, model) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_wrap(~model) + theme_bw() + theme(legend.position = "none")


ggplot(data %>% filter(deletion_rate >= 0.4, deletion_rate<=0.55, Experiment=="Experiment2", predictability_weight==1, Region=="V1_0") %>% mutate(model=paste(predictability_weight, deletion_rate, Script, ID)) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio, model) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_wrap(~model) + theme_bw() + theme(legend.position = "none")




ggplot(data %>% filter(deletion_rate >= 0.4, deletion_rate<=0.55, Experiment=="Experiment1", predictability_weight==1, Region=="V1_0") %>% mutate(model=paste(predictability_weight, deletion_rate, Script, ID)) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio, model) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_wrap(~model) + theme_bw() + theme(legend.position = "none")



# By deletion rate, raw

ggplot(data %>% filter(Experiment=="Experiment2", predictability_weight==1, Region=="V1_0") %>% mutate(model=paste(predictability_weight, deletion_rate, Script, ID)) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + theme_bw() + theme(legend.position = "none")

ggplot(data %>% filter(Experiment=="Experiment2", predictability_weight==1, Region=="V1_0") %>% mutate(model=paste(predictability_weight, deletion_rate, Script, ID)) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_wrap(~deletion_rate) + theme_bw() + theme(legend.position = "none")



ggplot(data %>% mutate(deletion_rate=round(10*deletion_rate)/10) %>% filter(Experiment=="Experiment2", predictability_weight==1, Region=="V1_0") %>% mutate(model=paste(predictability_weight, deletion_rate, Script, ID)) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_wrap(~deletion_rate) + theme_bw() + theme(legend.position = "none")

ggplot(data %>% mutate(deletion_rate=round(10*deletion_rate)/10) %>% filter(Experiment=="Experiment2", predictability_weight==1, Region=="V1_0") %>% mutate(model=paste(predictability_weight, deletion_rate, Script, ID)) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_wrap(~deletion_rate) + theme_bw() + theme(legend.position = "none")



ggplot(data %>% mutate(deletion_rate=round(10*deletion_rate)/10) %>% filter(Experiment=="Experiment2", Region=="V1_0") %>% mutate(model=paste(predictability_weight, deletion_rate, Script, ID)) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_wrap(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none")



for(experiment in c("Experiment1", "Experiment2")) {
   dataSmoothed = data.frame()
   params = unique(data %>% filter(predictability_weight==1) %>% select(deletion_rate, predictability_weight))
   for(i in (1:nrow(params))) {
   	del = params$deletion_rate[[i]]
   	pred = params$predictability_weight[[i]]
   	data_ = data %>% filter(predictability_weight==1, Region == "V1_0", Experiment == experiment, abs(deletion_rate-del) <= 0.05, abs(predictability_weight-pred) <= 0.25)
   	data_ = data_ %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))
   	data_$deletion_rate=del
   	data_$predictability_weight=pred
   	dataSmoothed = rbind(dataSmoothed, as.data.frame(data_))
   
   }
   
     plot = ggplot(dataSmoothed %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                   "TRUE_FALSE"="#00BA38",
                                   "TRUE_TRUE"="#619CFF")) 
   
   ggsave(plot, file=paste("figures/byParamRegion/conceptual-predictions-surprisal_VN_", experiment, "_OnlyPred_Smoothed.pdf", sep=""), width=10, height=1.5)

   plot = ggplot(dataSmoothed %>% filter(deletion_rate>0.33, deletion_rate<0.66) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                   "TRUE_FALSE"="#00BA38",
                                   "TRUE_TRUE"="#619CFF")) 
   
   ggsave(plot, file=paste("figures/byParamRegion/conceptual-predictions-surprisal_VN_", experiment, "_OnlyPred_Smoothed_MediumDeletionRate.pdf", sep=""), width=6, height=1.5)
}







for(experiment in c("Experiment1", "Experiment2")) {
   dataSmoothed = data.frame()
   params = unique(data %>% select(deletion_rate, predictability_weight))
   for(i in (1:nrow(params))) {
   	del = params$deletion_rate[[i]]
   	pred = params$predictability_weight[[i]]
   	data_ = data %>% filter(Region == "V2_0", Experiment == experiment, abs(deletion_rate-del) <= 0.05, abs(predictability_weight-pred) <= 0.25)
   	data_ = data_ %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))
   	data_$deletion_rate=del
   	data_$predictability_weight=pred
   	dataSmoothed = rbind(dataSmoothed, as.data.frame(data_))
   
   }
   
     plot = ggplot(dataSmoothed %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                   "TRUE_FALSE"="#00BA38",
                                   "TRUE_TRUE"="#619CFF")) 
   
   ggsave(plot, file=paste("figures/byParamRegion/conceptual-predictions-surprisal_VN_MiddelVerb_", experiment, "_Smoothed.pdf", sep=""), width=10, height=6)

   plot = ggplot(dataSmoothed %>% filter(deletion_rate>0.33, deletion_rate<0.66) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                   "TRUE_FALSE"="#00BA38",
                                   "TRUE_TRUE"="#619CFF")) 
   
   ggsave(plot, file=paste("figures/byParamRegion/conceptual-predictions-surprisal_VN_MiddleVerb_", experiment, "_Smoothed_MediumDeletionRate.pdf", sep=""), width=6, height=6)
}






plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC,  Condition, Noun, Ratio) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted)) %>% mutate(ThatFractionReweighted=ifelse(HasSC, ThatFractionReweighted, NA)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Posterior Belief Recovering 'that'") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/conceptual-predictions-that_VN.pdf", width=5, height=5)



plot = ggplot(data %>% filter(Experiment=="Experiment2", Region == "V2_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/predictions-surprisal_VN-MiddleVerb-Expt2.pdf", width=5, height=5)

plot = ggplot(data %>% filter(Experiment=="Experiment2", Region == "V2_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + theme_bw() + theme(legend.position = "none") + xlab("Embedding Bias") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 
ggsave(plot, file="figures/conceptual-predictions-surprisal_VN_MiddleVerb-Expt2.pdf", width=2, height=2)



