data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial.py.tsv", sep="\t")
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

data$compatible.C = (data$Condition == "SC_co")-0.5
summary(lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1|ID) + (1|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", deletion_rate==0.3, predictability_weight==0)))


model = (lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C +(1+compatible.C|ID) + (1+compatible.C|Item), data=data %>% filter(Region == "V1_0", deletion_rate==0.4, predictability_weight==0)))

byItemSlopes = coef(model)$Item
byItemSlopes$Item = rownames(byItemSlopes)

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



