library(tidyr)
library(dplyr)

data_E2 = read.csv("prepareMeansByExperiment_ByStimuli.R.tsv", quote='"', sep="\t") %>% filter(StimulusSet == "VN") %>% mutate(Experiment = "Experiment2")


data_E1 = read.csv("prepareMeansByExperiment_E1_ByStimuli.R.tsv", quote='"', sep="\t") %>% mutate(Experiment = "Experiment1", StimulusSet = "Experiment1")


data = rbind(data_E1, data_E2)

counts = unique(read.csv("../../../../../forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", sep="\t"))
counts$Ratio = log(counts$CountThat) - log(counts$CountBare)


scrc = read.csv("../../../../../forgetting/corpus_counts/wikipedia/RC_annotate/results/collectResults.py.tsv", sep="\t")
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

# the full raw predictions
plot = ggplot(data %>% filter(Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 

plot = ggplot(data %>% filter(Region == "V1_0") %>% filter(predictability_weight==1, deletion_rate>0.35, deletion_rate<0.6) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Experiment, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(~Experiment) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 


plot = ggplot(data %>% mutate(deletion_rate = round(deletion_rate*10)/10) %>% filter(Region == "V1_0") %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                "TRUE_FALSE"="#00BA38",
                                "TRUE_TRUE"="#619CFF")) 




   dataSmoothed = data.frame()
   params = unique(data %>% select(deletion_rate, predictability_weight))
   for(i in (1:nrow(params))) {
   	del = params$deletion_rate[[i]]
   	pred = params$predictability_weight[[i]]
   	data_ = data %>% filter(Region == "V1_0", abs(deletion_rate-del) <= 0.05, abs(predictability_weight-pred) <= 0.25)
   	data_ = data_ %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))
   	data_$deletion_rate=del
   	data_$predictability_weight=pred
   	dataSmoothed = rbind(dataSmoothed, as.data.frame(data_))
   
   }

   plot = ggplot(dataSmoothed %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                   "TRUE_FALSE"="#00BA38",
                                   "TRUE_TRUE"="#619CFF")) 
 

   plot = ggplot(dataSmoothed %>% filter(deletion_rate>0.33, deletion_rate<0.66) %>% group_by(compatible, HasSCHasRC, HasSC, HasRC, predictability_weight, deletion_rate, Condition, Noun, Ratio) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=HasSCHasRC)) + geom_smooth(method="lm", aes(linetype=compatible), se=F) + facet_grid(predictability_weight~deletion_rate) + theme_bw() + theme(legend.position = "none") + xlab("Log Embedding Rate") + ylab("Average Surprisal") + scale_color_manual(values = c("FALSE_FALSE" = "#F8766D",
                                   "TRUE_FALSE"="#00BA38",
                                   "TRUE_TRUE"="#619CFF")) 
 
