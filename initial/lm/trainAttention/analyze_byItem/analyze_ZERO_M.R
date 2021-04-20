data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_J_3_W_GPT2M_ZERO.py_332848174_ZeroLoss", sep="\t")
library(tidyr)
library(dplyr)
library(lme4)





nounFreqs = read.csv("../../../../../forgetting/corpus_counts/wikipedia/results/results_counts4.py.tsv", sep="\t")
nounFreqs$LCount = log(1+nounFreqs$Count)
nounFreqs$Condition = paste(nounFreqs$HasThat, nounFreqs$Capital, "False", sep="_")
nounFreqs = as.data.frame(unique(nounFreqs) %>% select(Noun, Condition, LCount) %>% group_by(Noun) %>% spread(Condition, LCount)) %>% rename(noun = Noun)


nounFreqs2 = read.csv("../../../../../forgetting/corpus_counts/wikipedia/results/archive/perNounCounts.csv") %>% mutate(X=NULL, ForgettingVerbLogOdds=NULL, ForgettingMiddleVerbLogOdds=NULL) %>% rename(noun = Noun) %>% mutate(True_True_True=NULL,True_False_True=NULL)

nounFreqs = unique(rbind(nounFreqs, nounFreqs2))
nounFreqs = nounFreqs[!duplicated(nounFreqs$noun),]

data = merge(data, nounFreqs %>% rename(Noun = noun), by=c("Noun"), all.x=TRUE)

data = data %>% mutate(True_Minus_False.C = True_False_False-False_False_False-mean(True_False_False-False_False_False, na.rm=TRUE))

unique((data %>% filter(is.na(True_Minus_False.C)))$Noun)
# [1] conjecture  guess       insinuation intuition   observation

data$compatible.C = (grepl("_co", data$Condition)-0.5)
data$HasRC.C = (grepl("SCRC", data$Condition)-0.5)
data$HasSC.C = (0.5-grepl("NoSC", data$Condition))

library(ggplot2)
plot = ggplot(data %>% group_by(Noun, True_Minus_False.C, Condition) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted)), aes(x=True_Minus_False.C, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + geom_text(aes(label=Noun))
ggsave(plot, file="figures/analyze_ZERO_M.R.pdf", height=8, width=8)



model = (lmer(SurprisalReweighted ~ HasRC.C * compatible.C + HasRC.C* True_Minus_False.C +  (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", HasSC.C > 0)))
library(ggrepel)
u = coef(model)$Item
u$Item = rownames(u)
plot = ggplot(u, aes(x=compatible.C)) + geom_histogram() + geom_text_repel(aes(label=Item, y=as.numeric(as.factor(Item))/5)) + theme_bw()
ggsave(plot, file="figures/analyze_ZERO_M.R_slopes_hist.pdf", height=8, width=8)
crash()


# Per-Item slopes u[order(u$compatible.C),]
#                              (Intercept)    HasRC.C compatible.C True_Minus_False.C
#o_bureaucrat_guard              10.961942 -0.1320201 -2.455005426          0.1076842
#v_psychiatrist_nurse            12.259260 -0.1320201 -2.448356952          0.1076842
#o_senator_diplomat               7.692690 -0.1320201 -2.412898858          0.1076842
#o_student_professor              8.670926 -0.1320201 -2.319344059          0.1076842
#o_lifeguard_swimmer             10.752732 -0.1320201 -2.023796876          0.1076842
#v_guest_cousin                  14.232131 -0.1320201 -1.957809447          0.1076842
#v_victim_swimmer                10.113895 -0.1320201 -1.734437068          0.1076842
#v_guest_thug                    11.390233 -0.1320201 -1.725085956          0.1076842
#v_thief_detective                9.559076 -0.1320201 -1.500484801          0.1076842
#v_driver_guide                   2.939419 -0.1320201 -1.480030577          0.1076842
#o_driver_tourist                 8.480536 -0.1320201 -1.223218894          0.1076842
#o_actor_star                     7.275376 -0.1320201 -1.138297229          0.1076842
#v_teacher_principal              3.359406 -0.1320201 -1.114128145          0.1076842
#o_preacher_parishioners         10.141369 -0.1320201 -1.039588785          0.1076842
#o_cousin_bror                   12.234769 -0.1320201 -0.984759337          0.1076842
#v_medic_survivor                 7.923781 -0.1320201 -0.828927733          0.1076842
#o_victim_criminal               14.918949 -0.1320201 -0.812472395          0.1076842
#v_sponsor_musician              11.332871 -0.1320201 -0.755545287          0.1076842

#> t.test(u[u$o,]$compatible.C)
#
#        One Sample t-test
#
#data:  u[u$o, ]$compatible.C
#t = -1.3847, df = 34, p-value = 0.1752
#alternative hypothesis: true mean is not equal to 0
#95 percent confidence interval:
# -0.6147648  0.1165016
#sample estimates:
# mean of x 
#-0.2491316 
#
#> t.test(u[!u$o,]$compatible.C)
#
#        One Sample t-test
#
#data:  u[!u$o, ]$compatible.C
#t = 0.35211, df = 31, p-value = 0.7271
#alternative hypothesis: true mean is not equal to 0
#95 percent confidence interval:
# -0.4412479  0.6254003
#sample estimates:
# mean of x 
#0.09207616 

