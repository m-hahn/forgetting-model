data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_J_3_W_GPT2XL_C_ZERO.py_883952439_ZeroLoss", sep="\t")
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



model = (lmer(SurprisalReweighted ~ HasRC.C * compatible.C + HasRC.C +  (1+compatible.C + HasRC.C + HasRC.C * compatible.C|Item), data=data %>% filter(Region == "V1_0", HasSC.C > 0, Noun == "report")))
#                     Estimate Std. Error t value
#(Intercept)           7.89722    0.47319  16.689
#HasRC.C              -0.12654    0.08386  -1.509
#compatible.C          0.03749    0.20962   0.179
#HasRC.C:compatible.C -0.28781    0.15470  -1.860


model = (lmer(SurprisalReweighted ~ HasRC.C * compatible.C + HasRC.C* True_Minus_False.C +  (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", HasSC.C > 0)))
#                            Estimate Std. Error t value
#(Intercept)                 7.677702   0.471715  16.276
#HasRC.C                    -0.200946   0.014148 -14.203
#compatible.C               -0.087176   0.189534  -0.460
#True_Minus_False.C         -0.080546   0.067784  -1.188
#HasRC.C:compatible.C       -0.230839   0.028296  -8.158
#HasRC.C:True_Minus_False.C  0.002202   0.016046   0.137


library(ggrepel)
u = coef(model)$Item
u$Item = rownames(u)
plot = ggplot(u, aes(x=compatible.C)) + geom_histogram() + geom_text_repel(aes(label=Item, y=as.numeric(as.factor(Item))/5)) + theme_bw()
ggsave(plot, file="figures/analyze_ZERO_L.R_slopes_hist.pdf", height=8, width=8)
write.table(u, file="analyze_ZERO_L.R.tsv", sep="\t")
crash()



# Per-Item slopes u[order(u$compatible.C),]
#                              (Intercept)    HasRC.C compatible.C True_Minus_False.C HasRC.C:compatible.C HasRC.C:True_Minus_False.C                          Item
#o_victim_criminal               16.443692 -0.2009456  -3.89012750        -0.08054613           -0.2308392                0.002202139             o_victim_criminal
#o_bureaucrat_guard              12.798415 -0.2009456  -3.40844048        -0.08054613           -0.2308392                0.002202139            o_bureaucrat_guard
#v_victim_swimmer                 9.588338 -0.2009456  -3.35275749        -0.08054613           -0.2308392                0.002202139              v_victim_swimmer
#o_preacher_parishioners         10.164426 -0.2009456  -2.61462249        -0.08054613           -0.2308392                0.002202139       o_preacher_parishioners
#v_driver_guide                   3.890123 -0.2009456  -2.51901611        -0.08054613           -0.2308392                0.002202139                v_driver_guide
#v_psychiatrist_nurse            13.366258 -0.2009456  -2.49010069        -0.08054613           -0.2308392                0.002202139          v_psychiatrist_nurse
#o_lifeguard_swimmer             12.053383 -0.2009456  -2.24142408        -0.08054613           -0.2308392                0.002202139           o_lifeguard_swimmer
#o_senator_diplomat               6.971846 -0.2009456  -2.20642963        -0.08054613           -0.2308392                0.002202139            o_senator_diplomat
#v_guest_cousin                  13.955848 -0.2009456  -1.97930941        -0.08054613           -0.2308392                0.002202139                v_guest_cousin
#o_bookseller_thief               8.399671 -0.2009456  -1.85934557        -0.08054613           -0.2308392                0.002202139            o_bookseller_thief
#v_president_farmer               8.077183 -0.2009456  -1.78781521        -0.08054613           -0.2308392                0.002202139            v_president_farmer
#v_actor_fans                    10.256402 -0.2009456  -1.28761903        -0.08054613           -0.2308392                0.002202139                  v_actor_fans
#v_guest_thug                     8.971182 -0.2009456  -1.13997745        -0.08054613           -0.2308392                0.002202139                  v_guest_thug
#v_medic_survivor                 7.195995 -0.2009456  -1.09813851        -0.08054613           -0.2308392                0.002202139              v_medic_survivor
#o_commander_president            9.904442 -0.2009456  -0.94565686        -0.08054613           -0.2308392                0.002202139         o_commander_president
#v_thief_detective                9.951851 -0.2009456  -0.86368388        -0.08054613           -0.2308392                0.002202139             v_thief_detective
#o_politician_banker              7.628845 -0.2009456  -0.80162446        -0.08054613           -0.2308392                0.002202139           o_politician_banker
#o_student_professor              7.817502 -0.2009456  -0.61356419        -0.08054613           -0.2308392                0.002202139           o_student_professor
#o_student_bully                 14.231308 -0.2009456  -0.58428238        -0.08054613           -0.2308392                0.002202139               o_student_bully
#o_tenant_foreman                 8.998755 -0.2009456  -0.51962206        -0.08054613           -0.2308392                0.002202139              o_tenant_foreman
#o_entrepreneur_philanthropist    5.453775 -0.2009456  -0.51561342        -0.08054613           -0.2308392                0.002202139 o_entrepreneur_philanthropist
#....
#o_neighbor_woman                 2.677498 -0.2009456   0.80525403        -0.08054613           -0.2308392                0.002202139              o_neighbor_woman
#v_bully_children                 3.297750 -0.2009456   0.82632835        -0.08054613           -0.2308392                0.002202139              v_bully_children
#o_trickster_woman               13.634632 -0.2009456   0.84976480        -0.08054613           -0.2308392                0.002202139             o_trickster_woman
#o_trader_businessman             2.799590 -0.2009456   0.88673417        -0.08054613           -0.2308392                0.002202139          o_trader_businessman
#o_clerk_customer                 8.095375 -0.2009456   0.90390935        -0.08054613           -0.2308392                0.002202139              o_clerk_customer
#o_criminal_officer               3.289029 -0.2009456   0.93283198        -0.08054613           -0.2308392                0.002202139            o_criminal_officer
#v_fiancé_author                  7.280652 -0.2009456   0.94354008        -0.08054613           -0.2308392                0.002202139               v_fiancé_author
#o_carpenter_craftsman            3.694351 -0.2009456   1.04216853        -0.08054613           -0.2308392                0.002202139         o_carpenter_craftsman
#v_banker_analyst                10.089040 -0.2009456   1.06726192        -0.08054613           -0.2308392                0.002202139              v_banker_analyst
#o_pharmacist_stranger           10.356985 -0.2009456   1.15479462        -0.08054613           -0.2308392                0.002202139         o_pharmacist_stranger
#o_child_medic                   11.721865 -0.2009456   1.16346285        -0.08054613           -0.2308392                0.002202139                 o_child_medic
#v_lifeguard_soldier             14.406856 -0.2009456   1.71025774        -0.08054613           -0.2308392                0.002202139           v_lifeguard_soldier
#o_actor_star                     6.941863 -0.2009456   1.77016498        -0.08054613           -0.2308392                0.002202139                  o_actor_star
#v_senator_diplomat              11.223623 -0.2009456   1.77497946        -0.08054613           -0.2308392                0.002202139            v_senator_diplomat
#o_surgeon_patient                3.261595 -0.2009456   1.79864220        -0.08054613           -0.2308392                0.002202139             o_surgeon_patient
#v_vendor_salesman               11.502253 -0.2009456   1.98733412        -0.08054613           -0.2308392                0.002202139             v_vendor_salesman
#o_mobster_media                  7.139858 -0.2009456   2.04248755        -0.08054613           -0.2308392                0.002202139               o_mobster_media
#v_captain_crew                   5.004972 -0.2009456   2.28463496        -0.08054613           -0.2308392                0.002202139                v_captain_crew
#v_agent_fbi                      4.570650 -0.2009456   3.25861236        -0.08054613           -0.2308392                0.002202139                   v_agent_fbi
#o_consultant_artist             10.277153 -0.2009456   4.69647943        -0.08054613           -0.2308392                0.002202139           o_consultant_artist

