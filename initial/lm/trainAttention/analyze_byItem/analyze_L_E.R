data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_J_3_W_GPT2L_SANI_E.py_682100515_Sanity", sep="\t")


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

model = (lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1+compatible.C|Noun), data=data %>% filter(Region == "V1_0")))

#                   Estimate Std. Error t value
#(Intercept)         9.16016    0.43591  21.014
#compatible.C        0.83833    0.19341   4.334
#True_Minus_False.C -0.22142    0.08109  -2.730



#                              (Intercept) compatible.C True_Minus_False.C                                                                                                                          [43/1910]
#o_bureaucrat_guard              13.170448  -3.15765872         -0.2214152
#v_guest_cousin                  14.192303  -2.44845498         -0.2214152
#o_lifeguard_swimmer             11.627661  -2.43951525         -0.2214152
#v_psychiatrist_nurse            12.006760  -2.11044203         -0.2214152
#v_driver_guide                   6.512957  -1.73820667         -0.2214152
#v_guest_thug                    12.252064  -1.63920287         -0.2214152
#o_commander_president           13.922310  -1.21941288         -0.2214152
#v_victim_swimmer                10.756746  -1.10650096         -0.2214152
#v_thief_detective               11.541931  -1.02214230         -0.2214152
#o_politician_banker              8.774737  -0.61294879         -0.2214152
#v_investor_scientist            13.577683  -0.55455480         -0.2214152
#o_victim_criminal               15.960820  -0.38624548         -0.2214152
#o_senator_diplomat               8.636443  -0.38243672         -0.2214152
#v_doctor_colleague               8.486581  -0.00494759         -0.2214152
#v_pediatrician_receptionist     12.043381   0.01779886         -0.2214152
#o_tenant_foreman                11.582135   0.01968767         -0.2214152
#o_musician_far                  11.941208   0.03379617         -0.2214152
#o_daughter_sister                9.895033   0.14028206         -0.2214152
#o_sculptor_painter               5.330139   0.15286600         -0.2214152
#o_preacher_parishioners         11.090458   0.18515791         -0.2214152
#o_actor_star                     7.958355   0.23150610         -0.2214152
#o_entrepreneur_philanthropist    9.027886   0.23958988         -0.2214152
#v_manager_boss                  10.664873   0.25202217         -0.2214152
#v_medic_survivor                 8.777386   0.34842193         -0.2214152
#o_driver_tourist                 8.370297   0.45985844         -0.2214152
#....
#v_plaintiff_jury                 9.243065   1.44021567         -0.2214152                                                                                                                           [0/1910]
#o_runner_psychiatrist            3.431785   1.49398434         -0.2214152
#o_child_medic                   13.807687   1.60884441         -0.2214152
#v_plumber_apprentice             4.653745   1.62372843         -0.2214152
#v_fiancé_author                  7.242115   1.67763736         -0.2214152
#o_mobster_media                  9.836199   1.76632206         -0.2214152
#o_neighbor_woman                 3.834489   1.80608332         -0.2214152
#v_sponsor_musician              11.879841   1.82749551         -0.2214152
#o_consultant_artist             12.583685   1.86656791         -0.2214152
#v_banker_analyst                11.424488   1.87306117         -0.2214152
#o_ceo_employee                   4.414571   2.01766711         -0.2214152
#v_lifeguard_soldier             14.701247   2.05259792         -0.2214152
#v_businessman_sponsor            7.896909   2.06705904         -0.2214152
#o_student_bully                 13.884633   2.09744706         -0.2214152
#v_fisherman_gardener             5.304991   2.15726710         -0.2214152
#v_vendor_salesman               11.919650   2.36858024         -0.2214152
#o_carpenter_craftsman            5.020058   2.43721374         -0.2214152
#v_senator_diplomat              11.673522   2.62952475         -0.2214152
#o_trader_businessman             5.011750   2.76940192         -0.2214152
#o_violinist_sponsors             5.165015   2.87292031         -0.2214152
#o_trickster_woman               16.326351   2.88928604         -0.2214152
#v_captain_crew                   6.763917   3.21761520         -0.2214152
#o_surgeon_patient                5.942820   3.51284487         -0.2214152
#o_criminal_officer               5.326119   4.05894652         -0.2214152
#v_agent_fbi                      5.518061   4.15759909         -0.2214152


# By-Noun Slopes:
#              (Intercept) compatible.C True_Minus_False.C
#feeling          9.741823    0.4177936         -0.2214152
#proof            8.360852    0.4191133         -0.2214152
#understanding    9.529070    0.4639147         -0.2214152
#conclusion       9.443306    0.4730475         -0.2214152
#evidence         8.407043    0.5370611         -0.2214152
#story            9.485410    0.5783192         -0.2214152
#view             9.804330    0.5785817         -0.2214152
#reminder        10.127584    0.5808251         -0.2214152
#truth            9.291727    0.5838973         -0.2214152
#allegation       9.011793    0.5859839         -0.2214152
#finding          9.074407    0.5987586         -0.2214152
#....
#news             8.534323    0.9104824         -0.2214152                                                                                                                                           [0/1991]
#statement        9.654439    0.9503078         -0.2214152
#assessment       9.170202    0.9528964         -0.2214152
#myth             9.480331    0.9835696         -0.2214152
#information      9.651851    1.0031885         -0.2214152
#disclosure       9.104447    1.0277830         -0.2214152
#admission        8.967819    1.0278701         -0.2214152
#declaration      9.028675    1.0554561         -0.2214152
#complaint        9.513692    1.0824682         -0.2214152
#speculation      8.953511    1.1035680         -0.2214152
#message          9.883579    1.1269405         -0.2214152
#conviction       9.745017    1.2059110         -0.2214152
#confirmation     8.683918    1.2569264         -0.2214152
#announcement     9.004865    1.2730288         -0.2214152
#prediction       9.016152    1.3115385         -0.2214152
#remark           9.008030    1.3570836         -0.2214152


###############################

model = (lmer(ThatFractionReweighted ~ compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0"))) 

#                              (Intercept) compatible.C True_Minus_False.C                                                                      
#o_violinist_sponsors             52.31230   -79.755547           8.159321
#o_consultant_artist              48.09138   -72.244807           8.159321
#o_criminal_officer               56.42378   -69.920680           8.159321
#v_manager_boss                   56.85608   -65.036191           8.159321
#o_student_bully                  51.20678   -61.527287           8.159321
#v_plumber_apprentice             57.22527   -61.260284           8.159321
#o_trader_businessman             63.81819   -59.901604           8.159321
#v_fisherman_gardener             39.51227   -59.181370           8.159321
#o_trickster_woman                51.38398   -58.272409           8.159321
#v_thief_detective                62.40122   -53.202811           8.159321
#o_carpenter_craftsman            51.46724   -51.035536           8.159321
#o_child_medic                    69.20896   -47.999528           8.159321                                                                                                                          [45/1867]
#o_runner_psychiatrist            70.87909   -46.422182           8.159321
#o_surgeon_patient                58.99882   -46.317774           8.159321
#v_banker_analyst                 61.06821   -46.290800           8.159321
#v_guest_thug                     57.23989   -45.963591           8.159321
#v_firefighter_neighbor           65.96999   -45.661243           8.159321
#o_ceo_employee                   66.58155   -45.260309           8.159321
#o_extremist_agent                51.55019   -43.892944           8.159321
#o_neighbor_woman                 70.77009   -43.406727           8.159321
#v_actor_fans                     50.00143   -41.108380           8.159321
#o_sculptor_painter               67.52520   -39.253680           8.159321
#v_driver_guide                   42.35070   -38.662615           8.159321
#.....
#v_judge_attorney                 76.68878   -12.005925           8.159321                                                                                                                           [1/1867]
#v_victim_swimmer                 69.99449   -11.645365           8.159321
#o_tenant_foreman                 42.32183   -11.275417           8.159321
#v_investor_scientist             73.08807   -11.180793           8.159321
#v_medic_survivor                 48.24778    -6.602153           8.159321
#o_bookseller_thief               74.18793    -5.816217           8.159321
#o_entrepreneur_philanthropist    73.69819    -2.852863           8.159321
#v_lifeguard_soldier              81.99817     7.696542           8.159321
#v_sponsor_musician               16.40821    10.471283           8.159321
#o_pharmacist_stranger            79.02628    11.688162           8.159321
#v_captain_crew                   54.58396    17.945413           8.159321
#v_vendor_salesman                63.70349    35.411317           8.159321



