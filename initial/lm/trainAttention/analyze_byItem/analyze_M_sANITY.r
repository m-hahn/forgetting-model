#data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial.py.tsv", sep="\t")
data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_J_3_W_GPT2M_SANI.py_594881340_Sanity", sep="\t")
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


# Interesting
u = data %>% group_by(compatible.C, Item, Condition) %>% summarise(surps = mean(SurprisalsWithoutThat - SurprisalsWithThat, na.rm=TRUE))                                                                  
summary(lm(surps ~ compatible.C, data=u))



crash()
summary(lmer(ThatFractionReweighted ~ HasRC.C*compatible.C + HasRC.C*True_Minus_False.C +  (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", HasSC.C>0)))
#                           Estimate Std. Error t value
#(Intercept)                 42.8683     2.3059  18.591
#HasRC.C                     -6.3382     0.3017 -21.008 - makes sense
#compatible.C                -5.5803     1.0474  -5.328 - makes sense
#True_Minus_False.C          21.4897     2.0042  10.722 - makes sense
#HasRC.C:compatible.C         0.3132     0.6034   0.519 - 
#HasRC.C:True_Minus_False.C  -1.0492     0.2760  -3.801 - makes sense


summary(lmer(SurprisalReweighted ~ HasRC.C*compatible.C + HasRC.C*True_Minus_False.C +  (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", HasSC.C>0)))
#                           Estimate Std. Error t value
#(Intercept)                 8.97848    0.40892  21.956
#HasRC.C                     0.05445    0.01245   4.373 - makes sense
#compatible.C               -0.09166    0.15475  -0.592
#True_Minus_False.C         -0.43558    0.06272  -6.945 - makes sense
#HasRC.C:compatible.C        0.20810    0.02490   8.356 - ! why? in surprisal but not ThatFractionReweighted??? Is it because the contexct is too short for the language model?
#HasRC.C:True_Minus_False.C -0.07009    0.01139  -6.153 - makes sense


summary(lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1|ID) + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", deletion_rate==0.55)))


model = (lmer(SurprisalReweighted ~ HasRC.C + compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", deletion_rate==0.35, HasSC.C>0, !grepl("GPT2M", Script))))
# The ordering of slopes closely replicates -- so it is a property of GPT2 and stable across multiple runs of the memory model
#                              (Intercept)     HasRC.C compatible.C True_Minus_False.C
#v_guest_thug                    12.205749 -0.02194668 -2.545766952         -0.2042718
#v_psychiatrist_nurse            12.665236 -0.02194668 -2.208907242         -0.2042718
#o_lifeguard_swimmer             12.590915 -0.02194668 -1.732583283         -0.2042718
#o_bureaucrat_guard              13.837014 -0.02194668 -1.475658775         -0.2042718
#o_student_bully                 15.162266 -0.02194668 -1.080967839         -0.2042718
#o_senator_diplomat               8.612687 -0.02194668 -1.002161147         -0.2042718
#v_victim_swimmer                12.308828 -0.02194668 -0.981938076         -0.2042718
#v_thief_detective               10.328938 -0.02194668 -0.805953009         -0.2042718
#o_child_medic                   13.670466 -0.02194668 -0.723813968         -0.2042718
#o_bookseller_thief              11.756582 -0.02194668 -0.650526839         -0.2042718
#o_ceo_employee                   4.019287 -0.02194668 -0.546799699         -0.2042718
#o_commander_president           14.216134 -0.02194668 -0.507111841         -0.2042718
#v_doctor_colleague               8.566765 -0.02194668 -0.448448220         -0.2042718
#o_trickster_woman               15.516596 -0.02194668 -0.266296919         -0.2042718

# Interesting difference between the VN and the other items (seems to replicate across configurations)
# u = coef(model)$Item
#> u$v = grepl("v_", u$item)
#> u1 = u[u$v,]
#> u2 = u[!u$v,]
#> t.test(u1$compatible.C)
#
#        One Sample t-test
#
#data:  u1$compatible.C
#t = 3.1617, df = 31, p-value = 0.003495
#alternative hypothesis: true mean is not equal to 0
#95 percent confidence interval:
# 0.2577982 1.1948335
#sample estimates:
#mean of x 
#0.7263159 
#
#> t.test(u2$compatible.C)
#
#        One Sample t-test
#
#data:  u2$compatible.C
#t = 1.2206, df = 34, p-value = 0.2306
#alternative hypothesis: true mean is not equal to 0
#95 percent confidence interval:
# -0.1278902  0.5125606
#sample estimates:
#mean of x 
#0.1923352 




model = (lmer(SurprisalReweighted ~ HasRC.C + compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", deletion_rate==0.55, HasSC.C>0)))
#                              (Intercept)   HasRC.C compatible.C True_Minus_False.C                                                 
#o_lifeguard_swimmer             12.502503 0.1911667  -2.79518307         -0.2625345
#v_psychiatrist_nurse            12.555227 0.1911667  -2.45026724         -0.2625345
#v_guest_thug                    13.251113 0.1911667  -2.36664022         -0.2625345
#o_daughter_sister                9.373278 0.1911667  -1.56298862         -0.2625345
#o_bureaucrat_guard              13.962854 0.1911667  -1.17707479         -0.2625345
#o_student_bully                 15.263687 0.1911667  -1.14062156         -0.2625345
#v_victim_swimmer                12.425828 0.1911667  -1.05488009         -0.2625345
#o_child_medic                   13.984062 0.1911667  -0.96919797         -0.2625345
#v_thief_detective               10.376001 0.1911667  -0.96217550         -0.2625345
#v_guest_cousin                  13.405787 0.1911667  -0.91731791         -0.2625345


summary(lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1|ID) + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", deletion_rate==0.3, predictability_weight==0)))

model = (lmer(SurprisalReweighted ~ deletion_rate + compatible.C + True_Minus_False.C + (1+compatible.C|ID) + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", predictability_weight==1)))

model = (lmer(SurprisalReweighted ~ predictability_weight + deletion_rate + compatible.C + True_Minus_False.C + (1+compatible.C|ID) + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0")))

summary(lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", predictability_weight==1) %>% group_by(Noun, compatible.C, True_Minus_False.C, Item) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted))))


model = (lmer(SurprisalReweighted ~ predictability_weight + deletion_rate + compatible.C + True_Minus_False.C + (1+compatible.C|ID) + (1+compatible.C|Item), data=data %>% filter(Region == "V1_0")))


byItemSlopes = coef(model)$Item
byItemSlopes$Item = rownames(byItemSlopes)
byItemSlopes[order(byItemSlopes$compatible.C),]


#> byItemSlopes[order(byItemSlopes$compatible.C),]                                                                                            
#                              (Intercept) predictability_weight deletion_rate compatible.C True_Minus_False.C                          Item
#o_child_medic                   7.4318434             -1.069783      8.131016  -3.20739170         -0.1758554                 o_child_medic
#o_senator_diplomat              4.8994153             -1.069783      8.131016  -3.18698116         -0.1758554            o_senator_diplomat
#o_mobster_media                 4.1932516             -1.069783      8.131016  -2.42209852         -0.1758554               o_mobster_media
#o_victim_criminal               8.6898748             -1.069783      8.131016  -2.17947177         -0.1758554             o_victim_criminal
#o_student_bully                 7.9275158             -1.069783      8.131016  -1.89616001         -0.1758554               o_student_bully
#v_psychiatrist_nurse            5.6874614             -1.069783      8.131016  -1.56520274         -0.1758554          v_psychiatrist_nurse
#o_lifesaver_swimmer             7.9101302             -1.069783      8.131016  -1.49021589         -0.1758554           o_lifesaver_swimmer
#v_guest_thug                    8.2118196             -1.069783      8.131016  -1.48537021         -0.1758554                  v_guest_thug
#o_CEO_employee                  1.9447643             -1.069783      8.131016  -1.30279823         -0.1758554                o_CEO_employee
#v_victim_swimmer                6.7159427             -1.069783      8.131016  -1.26434984         -0.1758554              v_victim_swimmer
#v_teacher_principal             0.7832282             -1.069783      8.131016  -0.96076600         -0.1758554           v_teacher_principal
#v_sponsor_musician              7.9813119             -1.069783      8.131016  -0.95908404         -0.1758554            v_sponsor_musician
#o_bureaucrat_guard             10.1710971             -1.069783      8.131016  -0.85949145         -0.1758554            o_bureaucrat_guard
#v_medic_survivor                4.9993047             -1.069783      8.131016  -0.85518928         -0.1758554              v_medic_survivor
#o_surgeon_patient               2.4059452             -1.069783      8.131016  -0.78521757         -0.1758554             o_surgeon_patient
#v_driver_guide                  3.4735911             -1.069783      8.131016  -0.68676289         -0.1758554                v_driver_guide
#v_janitor_organizer             2.3548375             -1.069783      8.131016  -0.63490585         -0.1758554           v_janitor_organizer
#o_sculptor_painter              3.2599189             -1.069783      8.131016  -0.62205515         -0.1758554            o_sculptor_painter
#v_investor_scientist            7.4722328             -1.069783      8.131016  -0.60643089         -0.1758554          v_investor_scientist
#v_doctor_colleague              4.4566625             -1.069783      8.131016  -0.54478152         -0.1758554            v_doctor_colleague
#v_thief_detective               4.9405932             -1.069783      8.131016  -0.48976081         -0.1758554             v_thief_detective
#o_pharmacist_stranger           6.6452869             -1.069783      8.131016  -0.46649846         -0.1758554         o_pharmacist_stranger
#o_cousin_bror                   7.5825574             -1.069783      8.131016  -0.43043542         -0.1758554                 o_cousin_bror
#o_daughter_sister               5.6936303             -1.069783      8.131016  -0.30288336         -0.1758554             o_daughter_sister
#o_commander_president           8.3226388             -1.069783      8.131016  -0.29558488         -0.1758554         o_commander_president
#v_pediatrician_receptionist     6.8678478             -1.069783      8.131016  -0.27311863         -0.1758554   v_pediatrician_receptionist
#o_musician_far                  6.2599674             -1.069783      8.131016  -0.21466778         -0.1758554                o_musician_far
#o_actor_starlet                 4.4613360             -1.069783      8.131016  -0.20599745         -0.1758554               o_actor_starlet
#v_actor_fans                    5.7156175             -1.069783      8.131016  -0.11907038         -0.1758554                  v_actor_fans
#v_president_farmer              4.3350909             -1.069783      8.131016  -0.11284931         -0.1758554            v_president_farmer
#v_customer_vendor               2.0140520             -1.069783      8.131016  -0.08439834         -0.1758554             v_customer_vendor
#o_tenant_foreman                7.6368918             -1.069783      8.131016   0.07788745         -0.1758554              o_tenant_foreman
#v_plaintiff_jury                4.1507159             -1.069783      8.131016   0.17515407         -0.1758554              v_plaintiff_jury
#v_criminal_stranger             4.6489168             -1.069783      8.131016   0.25419893         -0.1758554           v_criminal_stranger
#v_firefighter_neighbor          6.0258317             -1.069783      8.131016   0.30957481         -0.1758554        v_firefighter_neighbor
#o_extremist_agent               2.1334148             -1.069783      8.131016   0.34295117         -0.1758554             o_extremist_agent
#v_fisherman_gardener            2.4621326             -1.069783      8.131016   0.37617751         -0.1758554          v_fisherman_gardener
#v_plumber_apprentice            1.6586674             -1.069783      8.131016   0.49112117         -0.1758554          v_plumber_apprentice
#v_bully_children                1.1935835             -1.069783      8.131016   0.60191783         -0.1758554              v_bully_children
#o_student_professor             3.4573072             -1.069783      8.131016   0.60531945         -0.1758554           o_student_professor
#v_guest_cousin                  3.6737238             -1.069783      8.131016   0.63851850         -0.1758554                v_guest_cousin
#v_judge_attorney                0.5360451             -1.069783      8.131016   0.70388971         -0.1758554              v_judge_attorney
#v_vendor_salesman               6.5744689             -1.069783      8.131016   0.72010601         -0.1758554             v_vendor_salesman
#o_driver_tourist                6.1229162             -1.069783      8.131016   0.76131710         -0.1758554              o_driver_tourist
#v_manager_boss                  6.7336547             -1.069783      8.131016   0.76873938         -0.1758554                v_manager_boss
#v_lifeguard_soldier             9.3321709             -1.069783      8.131016   0.78086825         -0.1758554           v_lifeguard_soldier
#o_consultant_artist             7.3432777             -1.069783      8.131016   0.79827348         -0.1758554           o_consultant_artist
#v_captain_crew                  3.1820146             -1.069783      8.131016   0.84955351         -0.1758554                v_captain_crew
#v_fiancé_author                 3.5447566             -1.069783      8.131016   0.87871249         -0.1758554               v_fiancé_author
#v_banker_analyst                5.8870418             -1.069783      8.131016   0.92507940         -0.1758554              v_banker_analyst
#o_principal_teacher             2.5421321             -1.069783      8.131016   1.00835969         -0.1758554           o_principal_teacher
#o_scientist_mayor               1.9104177             -1.069783      8.131016   1.08710326         -0.1758554             o_scientist_mayor
#o_neighbor_woman                0.7105074             -1.069783      8.131016   1.10618938         -0.1758554              o_neighbor_woman
#o_bookseller_thief              6.1366239             -1.069783      8.131016   1.22605821         -0.1758554            o_bookseller_thief
#o_entrepreneur_philanthropist   4.3852640             -1.069783      8.131016   1.23242511         -0.1758554 o_entrepreneur_philanthropist
#o_runner_psychiatrist           3.2139144             -1.069783      8.131016   1.55997332         -0.1758554         o_runner_psychiatrist
#o_trickster_woman               9.1696962             -1.069783      8.131016   1.72648661         -0.1758554             o_trickster_woman
#o_clerk_customer                8.0349529             -1.069783      8.131016   1.74485865         -0.1758554              o_clerk_customer
#v_businessman_sponsor           4.1200810             -1.069783      8.131016   1.76230711         -0.1758554         v_businessman_sponsor
#o_carpenter_craftsman           2.3730616             -1.069783      8.131016   1.76627631         -0.1758554         o_carpenter_craftsman
#v_senator_diplomat              6.8134852             -1.069783      8.131016   1.79939330         -0.1758554            v_senator_diplomat
#o_violinist_sponsors            2.3170906             -1.069783      8.131016   1.84205586         -0.1758554          o_violinist_sponsors
#o_politician_banker             5.1382865             -1.069783      8.131016   1.85389681         -0.1758554           o_politician_banker
#o_preacher_parishioners         6.4162186             -1.069783      8.131016   2.25625818         -0.1758554       o_preacher_parishioners
#v_agent_fbi                     2.8422874             -1.069783      8.131016   2.52469306         -0.1758554                   v_agent_fbi
#o_trader_businessman            1.6695241             -1.069783      8.131016   3.05425309         -0.1758554          o_trader_businessman
#o_criminal_officer              3.4731695             -1.069783      8.131016   3.26119981         -0.1758554            o_criminal_officer


byIDSlopes = coef(model)$ID
byIDSlopes$ID = rownames(byIDSlopes)
byIDSlopes[order(byIDSlopes$compatible.C),]


#> byIDSlopes[order(byIDSlopes$compatible.C),]     
#          (Intercept) predictability_weight deletion_rate compatible.C True_Minus_False.C        ID
#493283383    3.422375             -1.069783      8.131016 -0.038928946         -0.1758554 493283383
#584015835    4.182614             -1.069783      8.131016 -0.037595456         -0.1758554 584015835
#922774826    5.041638             -1.069783      8.131016 -0.012694818         -0.1758554 922774826
#99767452     4.895182             -1.069783      8.131016 -0.005788567         -0.1758554  99767452
#961536309    5.059172             -1.069783      8.131016  0.008153394         -0.1758554 961536309
#553302187    4.616161             -1.069783      8.131016  0.012184127         -0.1758554 553302187
#193988359    4.630897             -1.069783      8.131016  0.059016278         -0.1758554 193988359
#345336356    4.707851             -1.069783      8.131016  0.069778601         -0.1758554 345336356
#191511088    4.877481             -1.069783      8.131016  0.114371368         -0.1758554 191511088
#992213137    4.823875             -1.069783      8.131016  0.123880359         -0.1758554 992213137
#675784233    4.414690             -1.069783      8.131016  0.124437922         -0.1758554 675784233
#464657019    4.991215             -1.069783      8.131016  0.133107835         -0.1758554 464657019
#94907627     4.970169             -1.069783      8.131016  0.145892245         -0.1758554  94907627
#444273729    4.083726             -1.069783      8.131016  0.149230489         -0.1758554 444273729
#79010925     5.793023             -1.069783      8.131016  0.156401155         -0.1758554  79010925
#767406753    4.745672             -1.069783      8.131016  0.172478313         -0.1758554 767406753
#278167740    5.511846             -1.069783      8.131016  0.182801120         -0.1758554 278167740
#116146778    4.570194             -1.069783      8.131016  0.195945208         -0.1758554 116146778
#282352930    4.850553             -1.069783      8.131016  0.205365500         -0.1758554 282352930
#637269688    4.741264             -1.069783      8.131016  0.210219717         -0.1758554 637269688
#591357781    4.594979             -1.069783      8.131016  0.210298996         -0.1758554 591357781
#954662806    5.228781             -1.069783      8.131016  0.211052906         -0.1758554 954662806
#708115795    5.516360             -1.069783      8.131016  0.217600266         -0.1758554 708115795
#936548541    4.858380             -1.069783      8.131016  0.224033626         -0.1758554 936548541
#465577363    5.359163             -1.069783      8.131016  0.229594459         -0.1758554 465577363
#95795388     5.790406             -1.069783      8.131016  0.231922422         -0.1758554  95795388
#179088476    5.236838             -1.069783      8.131016  0.235244000         -0.1758554 179088476
#681474707    5.969039             -1.069783      8.131016  0.249414018         -0.1758554 681474707
#908049000    5.110056             -1.069783      8.131016  0.265561308         -0.1758554 908049000
#498236788    4.879595             -1.069783      8.131016  0.266751106         -0.1758554 498236788
#991579562    5.325402             -1.069783      8.131016  0.275085402         -0.1758554 991579562
#250967824    5.098741             -1.069783      8.131016  0.298764890         -0.1758554 250967824
#73230605     5.517735             -1.069783      8.131016  0.309486367         -0.1758554  73230605
#788091576    5.486403             -1.069783      8.131016  0.313975632         -0.1758554 788091576
#174187200    4.203831             -1.069783      8.131016  0.427892275         -0.1758554 174187200



crash()

summary(lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1|ID) + (1|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", deletion_rate==0.3, predictability_weight==0)))


model = (lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C +(1+compatible.C|ID) + (1+compatible.C|Item), data=data %>% filter(Region == "V1_0", deletion_rate==0.4, predictability_weight==0)))

byItemSlopes = coef(model)$Item
byItemSlopes$Item = rownames(byItemSlopes)


# Interesting (but the evidence isn't strong, might be accidental):
cor.test(byItemSlopes[["(Intercept)"]], byItemSlopes$compatible.C)

# These slopes are very similar with predictability_weight==0.5
#                              (Intercept) compatible.C True_Minus_False.C                          Item
#o_child_medic                   10.768493  -3.89992285         -0.1648946                 o_child_medic  was unharmed
#o_senator_diplomat               8.241476  -3.16607292         -0.1648946            o_senator_diplomat  was winning
#o_mobster_media                  7.138935  -2.93748887         -0.1648946               o_mobster_media  had disappeared
#o_victim_criminal               11.847292  -2.25583172         -0.1648946             o_victim_criminal were surviving
#o_student_bully                 10.997521  -1.87912327         -0.1648946               o_student_bully plagiarized his homework/drove everyone crazy
#v_teacher_principal              4.395694  -1.84736038         -0.1648946           v_teacher_principal  failed the student/annoyed the student
#v_guest_thug                    11.881792  -1.71798860         -0.1648946                  v_guest_thug tricked the bartender/stunned the bartender
#o_lifesaver_swimmer             11.766760  -1.66632545         -0.1648946           o_lifesaver_swimmer saved the children/pleased the children
#v_victim_swimmer                10.106983  -1.55286668         -0.1648946              v_victim_swimmer
#v_psychiatrist_nurse             9.310187  -1.40517317         -0.1648946          v_psychiatrist_nurse
#o_surgeon_patient                5.661679  -1.32479741         -0.1648946             o_surgeon_patient
#v_driver_guide                   6.715940  -1.20077072         -0.1648946                v_driver_guide
#v_janitor_organizer              5.808140  -0.91743157         -0.1648946           v_janitor_organizer
#o_cousin_bror                   11.838276  -0.86757017         -0.1648946                 o_cousin_bror
#o_pharmacist_stranger            9.513158  -0.84421798         -0.1648946         o_pharmacist_stranger
#v_sponsor_musician              11.517677  -0.79750164         -0.1648946            v_sponsor_musician
#o_sculptor_painter               6.743033  -0.79153043         -0.1648946            o_sculptor_painter
#v_medic_survivor                 8.429735  -0.72173719         -0.1648946              v_medic_survivor
#o_CEO_employee                   4.625234  -0.66604694         -0.1648946                o_CEO_employee
#o_bureaucrat_guard              13.422069  -0.62727965         -0.1648946            o_bureaucrat_guard
#v_investor_scientist            10.957524  -0.52012056         -0.1648946          v_investor_scientist
#v_thief_detective                7.967355  -0.30616220         -0.1648946             v_thief_detective
#v_president_farmer               7.736207  -0.28583124         -0.1648946            v_president_farmer
#v_actor_fans                     9.068854  -0.25547948         -0.1648946                  v_actor_fans
#v_pediatrician_receptionist     10.050521  -0.21028115         -0.1648946   v_pediatrician_receptionist
#v_criminal_stranger              8.134394  -0.17040682         -0.1648946           v_criminal_stranger
#o_musician_far                   9.926035  -0.16965755         -0.1648946                o_musician_far
#o_actor_starlet                  7.833042  -0.13629062         -0.1648946               o_actor_starlet
#o_commander_president           11.969829  -0.13440355         -0.1648946         o_commander_president                                     
#o_daughter_sister                9.036913  -0.08991114         -0.1648946             o_daughter_sister
#v_doctor_colleague               8.161714  -0.04984468         -0.1648946            v_doctor_colleague
#v_customer_vendor                5.281899   0.09477463         -0.1648946             v_customer_vendor
#o_tenant_foreman                10.788711   0.14762019         -0.1648946              o_tenant_foreman
#o_consultant_artist             11.425500   0.20614368         -0.1648946           o_consultant_artist
#o_extremist_agent                5.144713   0.31047188         -0.1648946             o_extremist_agent
#v_plaintiff_jury                 7.536574   0.35119581         -0.1648946              v_plaintiff_jury
#v_fisherman_gardener             5.598490   0.36320928         -0.1648946          v_fisherman_gardener
#v_plumber_apprentice             4.920909   0.44212153         -0.1648946          v_plumber_apprentice
#v_judge_attorney                 3.931734   0.50723899         -0.1648946              v_judge_attorney
#v_firefighter_neighbor           9.436054   0.52512666         -0.1648946        v_firefighter_neighbor
#v_bully_children                 4.456294   0.53172195         -0.1648946              v_bully_children
#v_guest_cousin                   6.623165   0.60596852         -0.1648946                v_guest_cousin
#v_fiancé_author                  6.795485   0.63975496         -0.1648946               v_fiancé_author
#v_captain_crew                   6.619463   0.65467070         -0.1648946                v_captain_crew
#o_student_professor              6.678425   0.77189760         -0.1648946           o_student_professor
#o_driver_tourist                 9.482586   0.78555334         -0.1648946              o_driver_tourist
#o_neighbor_woman                 3.982770   0.81545676         -0.1648946              o_neighbor_woman
#v_manager_boss                  10.154127   0.83507588         -0.1648946                v_manager_boss
#o_clerk_customer                10.963351   0.88229165         -0.1648946              o_clerk_customer
#v_banker_analyst                 9.185154   0.91583154         -0.1648946              v_banker_analyst
#o_entrepreneur_philanthropist    7.932043   0.97989890         -0.1648946 o_entrepreneur_philanthropist
#o_principal_teacher              5.808110   1.27180075         -0.1648946           o_principal_teacher
#v_lifeguard_soldier             12.752973   1.35196404         -0.1648946           v_lifeguard_soldier
#o_scientist_mayor                5.418019   1.43988180         -0.1648946             o_scientist_mayor
#v_vendor_salesman               10.181121   1.45451247         -0.1648946             v_vendor_salesman
#v_senator_diplomat              10.054649   1.54162695         -0.1648946            v_senator_diplomat
#o_violinist_sponsors             5.692785   1.64969114         -0.1648946          o_violinist_sponsors
#v_businessman_sponsor            7.489190   1.70923178         -0.1648946         v_businessman_sponsor
#o_criminal_officer               6.020972   1.84912295         -0.1648946            o_criminal_officer
#o_politician_banker              8.400021   2.08268570         -0.1648946           o_politician_banker
#o_trickster_woman               12.605169   2.16073843         -0.1648946             o_trickster_woman
#o_runner_psychiatrist            5.962136   2.21337998         -0.1648946         o_runner_psychiatrist
#o_carpenter_craftsman            5.512902   2.26077326         -0.1648946         o_carpenter_craftsman
#o_preacher_parishioners          9.665387   2.35810326         -0.1648946       o_preacher_parishioners
#v_agent_fbi                      6.293955   2.43050818         -0.1648946                   v_agent_fbi	arrested the criminal/confused the criminal
#o_bookseller_thief               9.516952   2.86883164         -0.1648946            o_bookseller_thief	got a heart attack
#o_trader_businessman             4.887210   3.18120764         -0.1648946          o_trader_businessman	had insider information
#


model2 = (lmer(ThatFractionReweighted ~ compatible.C + True_Minus_False.C +(1+compatible.C|ID) + (1+compatible.C|Item), data=data %>% filter(Region == "V1_0", deletion_rate==0.4, predictability_weight==0)))

byItemSlopes = coef(model2)$Item
byItemSlopes$Item = rownames(byItemSlopes)

#o_scientist_mayor                87.46241 -13.11925475           2.807508             o_scientist_mayor         had faked data/couldn't be trusted                                            
#o_entrepreneur_philanthropist    83.79822 -10.55660954           2.807508 o_entrepreneur_philanthropist	wasted the money/exasperated the nurse
#o_criminal_officer               80.39496 -10.29351953           2.807508            o_criminal_officer	was guilty/was refuted
#o_surgeon_patient                85.04842 -10.15585118           2.807508             o_surgeon_patient	had no degree/was widely known
#o_preacher_parishioners          86.71091  -9.47752685           2.807508       o_preacher_parishioners
#o_trader_businessman             88.52969  -9.37560438           2.807508          o_trader_businessman
#o_extremist_agent                83.56492  -8.08450427           2.807508             o_extremist_agent
#v_plumber_apprentice             84.26964  -7.92866378           2.807508          v_plumber_apprentice
#o_violinist_sponsors             85.94343  -7.12730111           2.807508          o_violinist_sponsors
#v_banker_analyst                 85.28712  -7.05238195           2.807508              v_banker_analyst
#o_trickster_woman                83.10796  -7.03971028           2.807508             o_trickster_woman
#v_thief_detective                85.41960  -6.97343971           2.807508             v_thief_detective
#v_driver_guide                   84.91185  -6.97265389           2.807508                v_driver_guide
#o_sculptor_painter               83.74767  -6.87932938           2.807508            o_sculptor_painter
#o_child_medic                    84.87304  -6.81020907           2.807508                 o_child_medic
#o_cousin_bror                    82.94171  -6.55995474           2.807508                 o_cousin_bror
#v_fisherman_gardener             82.68731  -6.47733735           2.807508          v_fisherman_gardener
#v_senator_diplomat               84.04256  -6.43737907           2.807508            v_senator_diplomat
#v_firefighter_neighbor           82.77441  -6.39140299           2.807508        v_firefighter_neighbor
#v_guest_cousin                   80.44554  -6.36035586           2.807508                v_guest_cousin
#o_principal_teacher              85.09546  -6.14038844           2.807508           o_principal_teacher
#o_consultant_artist              79.50840  -6.08765829           2.807508           o_consultant_artist
#o_student_professor              84.82091  -5.74017191           2.807508           o_student_professor
#o_neighbor_woman                 87.04194  -5.44558344           2.807508              o_neighbor_woman
#o_bureaucrat_guard               82.47143  -5.08322115           2.807508            o_bureaucrat_guard
#v_janitor_organizer              84.11530  -5.04157705           2.807508           v_janitor_organizer
#o_bookseller_thief               82.66239  -4.93231882           2.807508            o_bookseller_thief
#o_lifesaver_swimmer              83.42137  -4.75432836           2.807508           o_lifesaver_swimmer
#v_customer_vendor                81.99634  -4.51363232           2.807508             v_customer_vendor
#o_mobster_media                  85.31739  -4.44291102           2.807508               o_mobster_media
#v_investor_scientist             82.73042  -4.22345472           2.807508          v_investor_scientist
#v_manager_boss                   83.33789  -4.22082904           2.807508                v_manager_boss
#v_guest_thug                     83.86669  -4.20451750           2.807508                  v_guest_thug
#v_bully_children                 84.65522  -4.15938218           2.807508              v_bully_children
#v_psychiatrist_nurse             86.60480  -3.95205088           2.807508          v_psychiatrist_nurse
#o_driver_tourist                 87.95997  -3.90034624           2.807508              o_driver_tourist
#v_pediatrician_receptionist      84.99795  -3.87422731           2.807508   v_pediatrician_receptionist
#o_commander_president            83.88214  -3.78209634           2.807508         o_commander_president
#o_runner_psychiatrist            83.35741  -3.48727869           2.807508         o_runner_psychiatrist
#o_actor_starlet                  84.07877  -3.33478506           2.807508               o_actor_starlet
#v_fiancé_author                  84.16542  -3.17837258           2.807508               v_fiancé_author
#v_actor_fans                     82.81152  -2.95411249           2.807508                  v_actor_fans
#o_senator_diplomat               82.65859  -2.90629655           2.807508            o_senator_diplomat
#v_plaintiff_jury                 86.25058  -2.90391998           2.807508              v_plaintiff_jury
#v_businessman_sponsor            89.41016  -2.62955707           2.807508         v_businessman_sponsor
#v_vendor_salesman                87.88541  -2.39227253           2.807508             v_vendor_salesman
#v_teacher_principal              85.68748  -2.27930926           2.807508           v_teacher_principal
#v_criminal_stranger              82.34367  -2.09696178           2.807508           v_criminal_stranger
#o_clerk_customer                 81.56801  -2.08883340           2.807508              o_clerk_customer
#v_victim_swimmer                 84.62578  -1.90370924           2.807508              v_victim_swimmer
#v_judge_attorney                 86.81673  -1.86969388           2.807508              v_judge_attorney
#o_carpenter_craftsman            84.02451  -1.69215045           2.807508         o_carpenter_craftsman
#o_pharmacist_stranger            87.53160  -1.40874657           2.807508         o_pharmacist_stranger
#o_daughter_sister                82.05274  -1.02103151           2.807508             o_daughter_sister
#v_medic_survivor                 81.97385  -0.97210659           2.807508              v_medic_survivor
#v_lifeguard_soldier              83.42388  -0.86695656           2.807508           v_lifeguard_soldier
#v_doctor_colleague               85.69208  -0.86127271           2.807508            v_doctor_colleague
#v_sponsor_musician               79.49338  -0.62902515           2.807508            v_sponsor_musician
#v_captain_crew                   83.89051  -0.10333422           2.807508                v_captain_crew
#o_student_bully                  78.81972   0.03648398           2.807508               o_student_bully
#o_victim_criminal                83.39799   0.11728546           2.807508             o_victim_criminal
#o_tenant_foreman                 81.54701   0.56521371           2.807508              o_tenant_foreman
#o_CEO_employee                   81.45328   0.74142469           2.807508                o_CEO_employee
#o_musician_far                   85.02372   1.34850899           2.807508                o_musician_far
#v_president_farmer               83.00137   1.89168617           2.807508            v_president_farmer
#v_agent_fbi                      84.32057   2.37295329           2.807508                   v_agent_fbi	arrested the criminal/confused the criminal
#o_politician_banker              78.68942   3.42211939           2.807508           o_politician_banker	laundered money/was popular

#
#   deletion_rate predictability_weight
#1           0.30                  0.00
#7           0.50                  0.00
#8           0.55                  0.00
#9           0.50                  0.25
#18          0.40                  0.00
#34          0.45                  0.00




model = (lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C +(1+compatible.C|ID) + (1+compatible.C|Item), data=data %>% filter(Region == "V1_0", deletion_rate==0.3, predictability_weight==0)))       


#                              (Intercept) compatible.C True_Minus_False.C                          Item                                                                                            [43/1866]
#o_mobster_media                  4.496106  -6.40501484        -0.05044264               o_mobster_media
#o_senator_diplomat               6.803331  -4.60917349        -0.05044264            o_senator_diplomat
#o_surgeon_patient                5.170188  -4.55348269        -0.05044264             o_surgeon_patient
#o_child_medic                    8.255213  -3.90376150        -0.05044264                 o_child_medic
#o_victim_criminal                8.915540  -3.39266871        -0.05044264             o_victim_criminal
#v_sponsor_musician              11.001467  -3.08297030        -0.05044264            v_sponsor_musician
#v_janitor_organizer              6.945259  -2.86266370        -0.05044264           v_janitor_organizer
#v_teacher_principal              3.672387  -2.46164142        -0.05044264           v_teacher_principal
#o_student_bully                  7.059583  -2.28519458        -0.05044264               o_student_bully
#v_driver_guide                   8.513113  -2.05352268        -0.05044264                v_driver_guide
#o_actor_starlet                  7.089057  -1.47102317        -0.05044264               o_actor_starlet
#o_CEO_employee                   5.201617  -1.36781963        -0.05044264                o_CEO_employee
#o_pharmacist_stranger            5.445108  -1.21810250        -0.05044264         o_pharmacist_stranger
#v_doctor_colleague               6.008020  -1.10516716        -0.05044264            v_doctor_colleague
#v_investor_scientist             8.868556  -0.99664887        -0.05044264          v_investor_scientist
#v_captain_crew                   7.767709  -0.95443300        -0.05044264                v_captain_crew
#o_lifesaver_swimmer              9.295139  -0.87443643        -0.05044264           o_lifesaver_swimmer
#o_cousin_bror                    8.042698  -0.85053028        -0.05044264                 o_cousin_bror
#v_president_farmer               6.877058  -0.85024550        -0.05044264            v_president_farmer
#o_musician_far                   6.591580  -0.84729952        -0.05044264                o_musician_far
#v_medic_survivor                 7.011600  -0.78314343        -0.05044264              v_medic_survivor
#v_psychiatrist_nurse             4.338750  -0.77113555        -0.05044264          v_psychiatrist_nurse
#v_plaintiff_jury                 6.952710  -0.74104999        -0.05044264              v_plaintiff_jury
#o_politician_banker              8.648524  -0.54187818        -0.05044264           o_politician_banker
#v_fiancé_author                  4.557252  -0.50849743        -0.05044264               v_fiancé_author
#o_bureaucrat_guard              14.123809  -0.49331308        -0.05044264            o_bureaucrat_guard
#o_daughter_sister                7.591207  -0.48653831        -0.05044264             o_daughter_sister
#v_victim_swimmer                 7.420335  -0.38961305        -0.05044264              v_victim_swimmer
#o_tenant_foreman                11.573853  -0.19512662        -0.05044264              o_tenant_foreman
#o_sculptor_painter               7.464312  -0.16894915        -0.05044264            o_sculptor_painter
#v_fisherman_gardener             7.562756  -0.10264386        -0.05044264          v_fisherman_gardener
#v_pediatrician_receptionist      7.905793  -0.03172616        -0.05044264   v_pediatrician_receptionist
#v_firefighter_neighbor           7.783144   0.17568348        -0.05044264        v_firefighter_neighbor
#v_actor_fans                     6.665987   0.22495957        -0.05044264                  v_actor_fans
#o_commander_president            8.774396   0.24140623        -0.05044264         o_commander_president
#v_thief_detective                6.095472   0.26015677        -0.05044264             v_thief_detective
#v_plumber_apprentice             4.651002   0.28086979        -0.05044264          v_plumber_apprentice
#v_customer_vendor                5.653264   0.39329909        -0.05044264             v_customer_vendor
#o_extremist_agent                4.553955   0.41307063        -0.05044264             o_extremist_agent
#v_banker_analyst                 6.972246   0.48150220        -0.05044264              v_banker_analyst
#v_vendor_salesman                7.713977   0.54299233        -0.05044264             v_vendor_salesman
#v_criminal_stranger              6.107583   0.55601081        -0.05044264           v_criminal_stranger
#v_lifeguard_soldier              8.272183   0.62926844        -0.05044264           v_lifeguard_soldier
#o_neighbor_woman                 3.474469   0.76624450        -0.05044264              o_neighbor_woman
#v_bully_children                 4.190889   0.81043300        -0.05044264              v_bully_children
#v_guest_thug                     9.103566   0.97714896        -0.05044264                  v_guest_thug
#v_judge_attorney                 3.392447   1.12898983        -0.05044264              v_judge_attorney
#v_manager_boss                   9.351564   1.68272760        -0.05044264                v_manager_boss
#o_carpenter_craftsman            6.782810   1.74324717        -0.05044264         o_carpenter_craftsman
#o_scientist_mayor                5.842819   1.79555457        -0.05044264             o_scientist_mayor
#v_guest_cousin                   9.372787   1.85162335        -0.05044264                v_guest_cousin
#o_principal_teacher              6.446017   1.87171343        -0.05044264           o_principal_teacher
#o_runner_psychiatrist            8.347469   2.00833295        -0.05044264         o_runner_psychiatrist
#o_driver_tourist                 7.971082   2.23188110        -0.05044264              o_driver_tourist
#v_agent_fbi                      6.750762   2.28444486        -0.05044264                   v_agent_fbi
#v_businessman_sponsor            6.073498   2.67452419        -0.05044264         v_businessman_sponsor
#o_student_professor              4.150456   2.72633119        -0.05044264           o_student_professor
#v_senator_diplomat               8.010948   2.81262533        -0.05044264            v_senator_diplomat
#o_violinist_sponsors             6.755304   3.07899238        -0.05044264          o_violinist_sponsors
#o_bookseller_thief               3.514508   3.23024444        -0.05044264            o_bookseller_thief
#o_clerk_customer                12.057934   3.62944288        -0.05044264              o_clerk_customer
#o_consultant_artist              7.842055   3.72479296        -0.05044264           o_consultant_artist
#o_trickster_woman                8.248278   4.07853995        -0.05044264             o_trickster_woman
#o_preacher_parishioners          7.460335   4.76061998        -0.05044264       o_preacher_parishioners
#o_criminal_officer               6.968645   4.93237544        -0.05044264            o_criminal_officer
#o_trader_businessman             4.644144   5.95040413        -0.05044264          o_trader_businessman
#o_entrepreneur_philanthropist    7.154386   7.32652036        -0.05044264 o_entrepreneur_philanthropist



data$Script.C = ifelse(data$Script == "script__J_3_W_GPT2M", -0.5, 0.5)

data$deletion_rate.C = data$deletion_rate-mean(data$deletion_rate, na.rm=TRUE)

model = (lmer(SurprisalReweighted ~ deletion_rate.C*compatible.C + Script.C*compatible.C + True_Minus_False.C+ (1|ID) + (1+compatible.C|Item), data=data %>% filter(Region == "V1_0", predictability_weight==0.5)))

byItemSlopes = coef(model)$Item
byItemSlopes$Item = rownames(byItemSlopes)


