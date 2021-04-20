data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_J_3_W_GPT2_ZERO.py_84881913_ZeroLoss", sep="\t")
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
ggsave(plot, file="figures/analyze_ZERO.R.pdf", height=8, width=8)



model = (lmer(SurprisalReweighted ~ HasRC.C * compatible.C + HasRC.C* True_Minus_False.C +  (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", HasSC.C > 0)))


library(ggrepel)
u = coef(model)$Item
u$Item = rownames(u)
plot = ggplot(u, aes(x=compatible.C)) + geom_histogram() + geom_text_repel(aes(label=Item, y=as.numeric(as.factor(Item))/5)) + theme_bw()
ggsave(plot, file="figures/analyze_ZERO.R_slopes_hist.pdf", height=8, width=8)
crash()



#
#                              (Intercept)    HasRC.C compatible.C True_Minus_False.C                                                                                                               [52/1992]
#v_guest_thug                    11.612591 -0.1245319 -2.937646174         -0.1319726
#v_psychiatrist_nurse            12.539127 -0.1245319 -2.074738144         -0.1319726
#o_lifeguard_swimmer             12.428458 -0.1245319 -1.251339079         -0.1319726
#o_bureaucrat_guard              13.780391 -0.1245319 -1.180715898         -0.1319726
#o_senator_diplomat               8.097895 -0.1245319 -1.100036010         -0.1319726
#o_student_bully                 14.946840 -0.1245319 -0.997941738         -0.1319726
#o_bookseller_thief              11.577815 -0.1245319 -0.873083852         -0.1319726
#v_victim_swimmer                11.773561 -0.1245319 -0.726377834         -0.1319726
#o_ceo_employee                   3.531573 -0.1245319 -0.712264496         -0.1319726
#v_thief_detective               10.189254 -0.1245319 -0.689263681         -0.1319726
#o_commander_president           13.778961 -0.1245319 -0.624453852         -0.1319726
#o_child_medic                   13.402635 -0.1245319 -0.543745781         -0.1319726
#o_victim_criminal               15.171121 -0.1245319 -0.443223131         -0.1319726
#v_doctor_colleague               8.186583 -0.1245319 -0.353593083         -0.1319726
#o_trickster_woman               15.131592 -0.1245319 -0.231738404         -0.1319726

