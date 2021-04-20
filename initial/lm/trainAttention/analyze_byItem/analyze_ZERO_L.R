data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_J_3_W_GPT2L_ZERO.py_32831323_ZeroLoss", sep="\t")
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
ggsave(plot, file="figures/analyze_ZERO_L.R.pdf", height=8, width=8)


model = (lmer(SurprisalReweighted ~ HasRC.C * compatible.C + HasRC.C* True_Minus_False.C +  (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", HasSC.C > 0)))
#Fixed effects:                                                                                                                  
#                            Estimate Std. Error t value
#(Intercept)                 8.684672   0.455793  19.054
#HasRC.C                    -0.226720   0.009226 -24.573
#compatible.C                0.210564   0.169274   1.244
#True_Minus_False.C         -0.001206   0.058716  -0.021
#HasRC.C:compatible.C       -0.214995   0.018453 -11.651
#HasRC.C:True_Minus_False.C -0.041138   0.008441  -4.874



library(ggrepel)
u = coef(model)$Item
u$Item = rownames(u)
plot = ggplot(u, aes(x=compatible.C)) + geom_histogram() + geom_text_repel(aes(label=Item, y=as.numeric(as.factor(Item))/5)) + theme_bw()
ggsave(plot, file="figures/analyze_ZERO_L.R_slopes_hist.pdf", height=8, width=8)
crash()



# Per-Item slopes u[order(u$compatible.C),]
#                              (Intercept)    HasRC.C compatible.C True_Minus_False.C HasRC.C:compatible.C HasRC.C:True_Minus_False.C                                               
#o_bureaucrat_guard              12.913126 -0.1800649  -3.59361647         -0.1791867           -0.2242652                 0.00950861
#v_driver_guide                   4.592052 -0.1800649  -2.67680314         -0.1791867           -0.2242652                 0.00950861
#v_guest_thug                    11.264083 -0.1800649  -2.62397397         -0.1791867           -0.2242652                 0.00950861
#v_guest_cousin                  14.698953 -0.1800649  -2.54186839         -0.1791867           -0.2242652                 0.00950861
#v_psychiatrist_nurse            11.931544 -0.1800649  -2.26196304         -0.1791867           -0.2242652                 0.00950861
#v_victim_swimmer                10.177526 -0.1800649  -2.08973761         -0.1791867           -0.2242652                 0.00950861
#o_lifeguard_swimmer             10.865607 -0.1800649  -1.91340769         -0.1791867           -0.2242652                 0.00950861
#o_commander_president           13.471081 -0.1800649  -1.52694670         -0.1791867           -0.2242652                 0.00950861
#o_politician_banker              8.702205 -0.1800649  -1.23774443         -0.1791867           -0.2242652                 0.00950861
#.......
#o_child_medic                   13.525293 -0.1800649   1.20973521         -0.1791867           -0.2242652                 0.00950861
#v_firefighter_neighbor          10.540565 -0.1800649   1.38420479         -0.1791867           -0.2242652                 0.00950861
#v_businessman_sponsor            7.799484 -0.1800649   1.53923469         -0.1791867           -0.2242652                 0.00950861
#v_sponsor_musician              11.185728 -0.1800649   1.56771089         -0.1791867           -0.2242652                 0.00950861
#o_carpenter_craftsman            4.256079 -0.1800649   1.58127995         -0.1791867           -0.2242652                 0.00950861
#o_student_bully                 13.500579 -0.1800649   1.58286904         -0.1791867           -0.2242652                 0.00950861
#v_lifeguard_soldier             14.371766 -0.1800649   1.80567240         -0.1791867           -0.2242652                 0.00950861
#v_senator_diplomat              11.492689 -0.1800649   2.01609549         -0.1791867           -0.2242652                 0.00950861
#v_vendor_salesman               12.504400 -0.1800649   2.06845695         -0.1791867           -0.2242652                 0.00950861
#o_criminal_officer               4.306611 -0.1800649   2.12128782         -0.1791867           -0.2242652                 0.00950861
#o_trickster_woman               15.635801 -0.1800649   2.42507581         -0.1791867           -0.2242652                 0.00950861
#o_surgeon_patient                5.334806 -0.1800649   2.64915028         -0.1791867           -0.2242652                 0.00950861
#v_captain_crew                   6.607844 -0.1800649   3.18019491         -0.1791867           -0.2242652                 0.00950861
#v_agent_fbi                      4.992108 -0.1800649   3.59416046         -0.1791867           -0.2242652                 0.00950861


