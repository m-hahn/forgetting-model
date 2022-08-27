data = read.csv("gpt3.tsv", sep="\t", header=F)
names(data) <- c("item", "HasRC", "compatible", "LogProb", "Word")

library(lme4)

data$HasRCF = data$HasRC
data$compatibleF = data$compatible

data$HasRC = (data$HasRC == "SCRC")
data$HasRC.C = data$HasRC - mean(data$HasRC)

data$compatible = (data$compatible == "compatible")
data$compatible.C = data$compatible - mean(data$compatible)


data$Surprisal = -data$LogProb

model = (lmer(Surprisal ~ HasRC.C + compatible.C + HasRC.C*compatible.C +  (1+HasRC.C+compatible.C|item) + (1+HasRC.C+compatible.C|Word), data=data))
#                     Estimate Std. Error t value
#(Intercept)            8.9223     0.5709  15.629
#HasRC.C                0.4211     0.1479   2.847
#compatible.C           0.2553     0.2212   1.154
#HasRC.C:compatible.C   0.2225     0.1648   1.350

model = (lmer(Surprisal ~ HasRC.C + compatible.C + HasRC.C*compatible.C +  (1+HasRC.C+compatible.C|item), data=data))
#                     Estimate Std. Error t value
#(Intercept)            6.8263     0.4029  16.943
#HasRC.C                0.4724     0.1325   3.566
#compatible.C           0.4150     0.1737   2.389
#HasRC.C:compatible.C   0.2225     0.2648   0.840


u = coef(model)$item
u[order(u$compatible.C),]
#                            (Intercept)   HasRC.C compatible.C
#psychiatrist_nurse            11.682785 0.4122684  -1.38694940
#bureaucrat_guard              10.126439 0.4315442  -1.04362203
#driver_tourist                 6.175343 0.4804796  -0.45536606
#victim_swimmer                 9.663053 0.4372834  -0.37376990
#driver_guide                   3.674021 0.5114591  -0.30362126
#actor_fans                     8.818393 0.4477447  -0.28416123
#guest_cousin                  11.593465 0.4133747  -0.23794764
#bookseller_thief               6.743742 0.4734398  -0.23027711
#investor_scientist            11.791348 0.4109239  -0.19715554
#politician_banker              5.301480 0.4913026  -0.13089980
#guest_thug                    10.056341 0.4324124  -0.10568214
#manager_boss                   6.988164 0.4704126  -0.09758000
#president_farmer               4.527632 0.5008869   0.03422970
#pharmacist_stranger            8.688041 0.4493591   0.03605335
#plaintiff_jury                 7.351213 0.4659161   0.03984972
#sponsor_musician               9.867343 0.4347532   0.04161111
#judge_attorney                 2.865685 0.5214706   0.06040893
#customer_vendor                4.152748 0.5055300   0.09422267
#fiancé_author                  7.009223 0.4701518   0.10231095
#pediatrician_receptionist     10.774893 0.4235129   0.12650530
#cousin_brother                10.586907 0.4258412   0.15239065
#entrepreneur_philanthropist    4.865410 0.4967035   0.17937738
#plumber_apprentice             3.225423 0.5170151   0.21999777
#musician_father               10.908437 0.4218589   0.22428805
#firefighter_neighbor           8.483047 0.4518981   0.28905356
#lifeguard_swimmer             10.274841 0.4297062   0.29898578
#medic_survivor                 7.867561 0.4595210   0.30884593
#vendor_salesman               10.982679 0.4209394   0.31191983
#banker_analyst                 8.527321 0.4513497   0.34549736
#bully_children                 3.221948 0.5170582   0.36529003
#victim_criminal               11.592118 0.4133914   0.37064824
#tenant_foreman                 9.344673 0.4412266   0.37245876
#violinist_sponsors             2.650757 0.5241325   0.37884065
#fisherman_gardener             2.868605 0.5214344   0.38588820
#preacher_parishioners          8.912877 0.4465745   0.39404253
#businessman_sponsor            4.863340 0.4967291   0.42358543
#captain_crew                   5.161202 0.4930400   0.43563244
#clerk_customer                 7.052496 0.4696158   0.46161781
#thief_detective                9.405071 0.4404785   0.51725370
#mobster_media                  6.343552 0.4783963   0.51910710
#daughter_sister                6.794527 0.4728108   0.56139876
#teacher_principal              2.405502 0.5271701   0.58462693
#scientist_mayor                2.013464 0.5320256   0.58935495
#doctor_colleague               6.087738 0.4815646   0.61444843
#senator_diplomat               8.908753 0.4466256   0.62830781
#carpenter_craftsman            3.575282 0.5126820   0.64271012
#trickster_woman               10.589895 0.4258042   0.65130085
#runner_psychiatrist            2.174744 0.5300281   0.70362204
#criminal_stranger              7.501305 0.4640572   0.74590998
#student_professor              6.397223 0.4777315   0.77644726
#sculptor_painter               2.656003 0.5240676   0.79337877
#commander_president            9.366309 0.4409586   0.82010122
#neighbor_woman                 2.475391 0.5263045   0.83408193
#janitor_organizer              3.521109 0.5133530   0.86879537
#principal_teacher              2.832351 0.5218835   0.87730654
#actor_star                     5.205290 0.4924940   0.88105277
#student_bully                 11.972684 0.4086780   0.92085043
#extremist_agent                3.842520 0.5093722   0.95810355
#trader_businessman             2.520125 0.5257505   0.95879689
#criminal_officer               3.518469 0.5133857   1.01110809
#child_medic                    8.667245 0.4496167   1.10043562
#ceo_employee                   2.895445 0.5211020   1.15214248
#lifeguard_soldier             13.349948 0.3916202   1.43998810
#surgeon_patient                3.643673 0.5118350   1.68299269
#agent_fbi                      4.809314 0.4973982   1.72976933
#consultant_artist              7.819548 0.4601157   2.22251251


# model = (lmer(Surprisal ~ HasRC.C + compatible.C + HasRC.C*compatible.C +  (1+HasRC.C+compatible.C|item) + (1+HasRC.C+compatible.C|Word), data=data))
# u = coef(model)$Word

#> cor.test(u$HasRC.C, u$compatible.C)
#
#	Pearson's product-moment correlation
#
#data:  u$HasRC.C and u$compatible.C
#t = -8.8332, df = 22, p-value = 1.097e-08                     !!!!!!!!!!!!!!!!!!!! what does this mean? maybe some identifiability issue?
#alternative hypothesis: true correlation is not equal to 0
#95 percent confidence interval:
# -0.9486252 -0.7453756
#sample estimates:
#       cor
#-0.8832076


#           (Intercept)      HasRC.C compatible.C HasRC.C:compatible.C
#relieved      8.562513 -0.249285579   1.07569554            0.2225332
#excited      10.783264 -0.217929128   0.82297199            0.2225332
#turned        7.194515 -0.076516542   1.00635053            0.2225332
#made          6.129208 -0.041342881   1.06875671            0.2225332
#devastated   11.641969 -0.002117832   0.48663487            0.2225332
#pleased      11.499631  0.009101020   0.48733593            0.2225332
#troubled     10.055548  0.186302923   0.42019793            0.2225332
#taught       11.635349  0.191635103   0.26030011            0.2225332
#calmed       11.277953  0.285036076   0.18564264            0.2225332
#confused     10.786963  0.288625336   0.22919146            0.2225332
#struck        8.565805  0.358368583   0.36351785            0.2225332
#seemed        6.825818  0.448523737   0.42713332            0.2225332
#did           7.249836  0.452363081   0.38139563            0.2225332
#is            3.226762  0.477144560   0.74364818            0.2225332
#sounded       9.456175  0.487189838   0.12600839            0.2225332
#was           3.225148  0.528174592   0.68402421            0.2225332
#shocked       7.991607  0.632919959   0.09773155            0.2225332
#proved        8.974463  0.691943307  -0.06700582            0.2225332
#annoyed      10.283169  0.771054256  -0.28696772            0.2225332
#startled     10.561430  0.788550128  -0.33452743            0.2225332
#deserved     12.567529  0.790791367  -0.53226597            0.2225332
#came          5.153770  0.798870179   0.17933105            0.2225332
#appeared      9.389290  1.209451405  -0.71360468            0.2225332
#impressed    11.096424  1.297806027  -0.98314634            0.2225332



library(ggplot2)
plot = ggplot(data, aes(x=Surprisal, color=paste(HasRCF, compatibleF), group = paste(HasRCF, compatibleF))) + geom_density()

u$item = rownames(u)

gpt2 = read.csv("gpt2.tsv", sep="\t", quot='"')

gpt2 = merge(gpt2, u, by=c("item"))

cor.test(gpt2$compatible.C.x, gpt2$compatible.C.y)


