data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial_QC.py.tsv", sep="\t")
#data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_Q_3_W_GPT2M.py_620912032_Model", sep="\t")
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
crash()

summary(lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1|ID) + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", deletion_rate==0.55)))


model = (lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1|ID) + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0")))


model = (lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", ID == 118571863, HasRC.C>0))) # Here: ThatFractionReweighted ~ 99.43
#                   Estimate Std. Error t value
#(Intercept)         7.58546    0.43661  17.374
#compatible.C       -0.30663    0.16318  -1.879
#True_Minus_False.C -0.12027    0.04131  -2.911

model = (lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", ID == 340958261, HasSC.C>0, HasRC.C < 0))) 
#                   Estimate Std. Error t value
#(Intercept)          8.9891     0.4202  21.393
#compatible.C         0.4715     0.1778   2.652
#True_Minus_False.C  -0.5431     0.1186  -4.581

model = (lmer(SurprisalReweighted ~ HasRC.C*compatible.C + HasRC.C*True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", ID == 340958261, HasSC.C>0))) 
#                           Estimate Std. Error t value
#(Intercept)                 9.18245    0.39704  23.127
#HasRC.C                     0.38470    0.02220  17.328
#compatible.C                0.56867    0.16674   3.411
#True_Minus_False.C         -0.44125    0.09672  -4.562
#HasRC.C:compatible.C        0.18915    0.04417   4.282
#HasRC.C:True_Minus_False.C  0.20273    0.02098   9.661


model = (lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", ID == 340958261, HasRC.C>0))) # ThatFraction ~ 34.9
#                   Estimate Std. Error t value
#(Intercept)         9.37571    0.38057  24.636
#compatible.C        0.66587    0.17711   3.760
#True_Minus_False.C -0.33941    0.08431  -4.026

#                              (Intercept) compatible.C True_Minus_False.C                                                                                                                           [48/453]
#o_principal_teacher              6.735914 -2.262172729         -0.3394132
#o_lifeguard_swimmer             12.201851 -2.258745732         -0.3394132
#v_psychiatrist_nurse            12.306987 -1.702843800         -0.3394132
#o_daughter_sister               10.898761 -1.609836991         -0.3394132
#v_driver_guide                   5.544647 -1.405544975         -0.3394132
#v_medic_survivor                 7.613208 -1.284630744         -0.3394132
#o_senator_diplomat               9.509173 -1.028021524         -0.3394132
#o_bureaucrat_guard              10.924730 -0.986881564         -0.3394132
#o_child_medic                   12.884394 -0.985190413         -0.3394132
#v_guest_cousin                  14.219785 -0.927290547         -0.3394132
#o_student_professor             10.838067 -0.913494778         -0.3394132
#v_victim_swimmer                11.230104 -0.708113866         -0.3394132
#o_actor_star                     8.103389 -0.687363044         -0.3394132
#o_driver_tourist                 8.977362 -0.548959795         -0.3394132
#v_guest_thug                    12.480916 -0.473464784         -0.3394132
#o_pharmacist_stranger           12.644455 -0.416350798         -0.3394132
#v_president_farmer               8.314064 -0.259655132         -0.3394132
#v_thief_detective                9.411520 -0.203799076         -0.3394132
#....
#o_ceo_employee                   5.075193  1.649162713         -0.3394132
#o_carpenter_craftsman            6.097640  1.732554882         -0.3394132
#v_plumber_apprentice             6.749909  1.919202481         -0.3394132
#o_neighbor_woman                 5.364507  2.066117961         -0.3394132
#v_banker_analyst                10.813736  2.107884996         -0.3394132
#v_fisherman_gardener             6.947918  2.117190298         -0.3394132
#v_captain_crew                   5.596918  2.161535342         -0.3394132
#v_fiancé_author                  8.526400  2.329782471         -0.3394132
#o_extremist_agent                6.931134  2.595982083         -0.3394132
#v_judge_attorney                 5.237604  2.625143077         -0.3394132
#v_agent_fbi                      7.120727  2.648860653         -0.3394132
#v_bully_children                 6.151890  2.684320617         -0.3394132
#o_trader_businessman             5.781854  2.722501190         -0.3394132
#o_surgeon_patient                6.932951  3.301127358         -0.3394132
#v_janitor_organizer              6.457288  3.343189793         -0.3394132
#o_criminal_officer               7.189620  4.707576793         -0.3394132

