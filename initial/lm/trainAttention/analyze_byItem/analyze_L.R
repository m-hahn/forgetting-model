data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/collect12_NormJudg_Short_Cond_W_GPT2_ByTrial.py.tsv", sep="\t", header=FALSE)
library(tidyr)
library(dplyr)
library(lme4)
names(data) <- c("Noun", "Item", "Region", "Condition", "Surprisal", "SurprisalReweighted", "ThatFraction", "ThatFractionReweighted", "SurpWithThat", "SurpWithoutThat", "Script", "ID", "predictability_weight", "deletion_rate", "autoencoder", "lm")
data = data %>% filter(grepl("GPT2L", Script))



nounFreqs = read.csv("../../../../../forgetting/corpus_counts/wikipedia/results/results_counts4.py.tsv", sep="\t")
nounFreqs$LCount = log(1+nounFreqs$Count)
nounFreqs$Condition = paste(nounFreqs$HasThat, nounFreqs$Capital, "False", sep="_")
nounFreqs = as.data.frame(unique(nounFreqs) %>% select(Noun, Condition, LCount) %>% group_by(Noun) %>% spread(Condition, LCount)) %>% rename(noun = Noun)


nounFreqs2 = read.csv("../../../../../forgetting/corpus_counts/wikipedia/results/archive/perNounCounts.csv") %>% mutate(X=NULL, ForgettingVerbLogOdds=NULL, ForgettingMiddleVerbLogOdds=NULL) %>% rename(noun = Noun) %>% mutate(True_True_True=NULL,True_False_True=NULL)

nounFreqs = unique(rbind(nounFreqs, nounFreqs2))
nounFreqs = nounFreqs[!duplicated(nounFreqs$noun),]

data = merge(data, nounFreqs %>% rename(Noun = noun), by=c("Noun"), all.x=TRUE)

data = data %>% mutate(True_Minus_False.C = True_False_False-False_False_False-mean(True_False_False-False_False_False, na.rm=TRUE))
data = data %>% mutate(Ratio = True_False_False-False_False_False)


library(ggplot2)

plot = ggplot(data %>% group_by(Condition, deletion_rate, Ratio, predictability_weight) %>% summarise(SurprisalReweighted=mean(SurprisalReweighted, na.rm=TRUE)), aes(x=Ratio, y=SurprisalReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate)
ggsave(plot, file="figures/analyze_L.R_surp.pdf")


plot = ggplot(data %>% group_by(Condition, deletion_rate, Ratio, predictability_weight) %>% summarise(ThatFractionReweighted=mean(ThatFractionReweighted, na.rm=TRUE)), aes(x=Ratio, y=ThatFractionReweighted, group=Condition, color=Condition)) + geom_smooth(method="lm") + facet_grid(predictability_weight~deletion_rate)
ggsave(plot, file="figures/analyze_L.R_that.pdf")


crash()




unique((data %>% filter(is.na(True_Minus_False.C)))$Noun)
# [1] conjecture  guess       insinuation intuition   observation

data$compatible.C = (grepl("_co", data$Condition)-0.5)
data$HasRC.C = (grepl("SCRC", data$Condition)-0.5)
data$HasSC.C = (0.5-grepl("NoSC", data$Condition))
crash()

summary(lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1|ID) + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", deletion_rate==0.5, predictability_weight=0.0)))

data2 = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_J_3_W_GPT2L_ZERO.py_32831323_ZeroLoss", sep="\t")

data = merge(data, data2 %>% group_by(Noun, Item, Region, Condition) %>% summarise(SurprisalZero = mean(SurprisalReweighted)), by=c("Noun", "Item", "Region", "Condition"))


for(id in unique(data$ID)) {
 model = (lmer(SurprisalReweighted ~ SurprisalZero + compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(ID == id))) 
 predictability_weight = unique(((data %>% filter(ID == id))$predictability_weight))[1]
 deletion_rate = unique(((data %>% filter(ID == id))$deletion_rate))[1]
 print(paste(id, " ", summary(model)$coef[3,3], " ", deletion_rate, " ", predictability_weight))
}

#[1] "599568962   1.52510761213343   0.5   0.75"
#[1] "261611831   1.41519779435684   0.5   0"
#[1] "620654642   2.76696949310314   0.55   1"
#[1] "68680906   1.63947812783625   0.55   1"
#[1] "3535245   3.07547998660363   0.55   0.75"
#[1] "185553481   1.86707832162419   0.55   0.75"
#[1] "978406179   0.83881763121371   0.45   0"
#[1] "795928457   1.08905389338733   0.5   0.75"
#[1] "262926142   -0.70113064005367   0.45   0.5"
#[1] "69019020   1.99082049550013   0.45   0.75"
#[1] "848128467   2.12594971742125   0.45   1"
#[1] "387749835   1.93933525385585   0.55   1"
#[1] "319356587   1.01738429658323   0.45   1"
#[1] "348524596   0.860003120551236   0.5   0"
#[1] "192234817   1.47062769791564   0.55   0.75"
#[1] "96253495   0.572872285120205   0.45   0.75"
#[1] "266998840   4.03082312583201   0.45   1"
#[1] "167196287   3.55475426704156   0.5   0.5"
#[1] "374361203   0.120816300968543   0.5   0.5"
#[1] "983021535   2.87143278426597   0.45   0.25"
#[1] "558918113   3.38323440262849   0.45   0.25"
#[1] "198947364   0.382985628300918   0.45   0"
#[1] "798647547   1.38514336709713   0.45   0.25"
#[1] "781374715   2.94334024490162   0.5   0.5"



zero_slopes = read.csv("analyze_ZERO_L.R.tsv", sep="\t")

data = merge(zero_slopes %>% rename(compatibleSlope = compatible.C) %>% select(Item, compatibleSlope), data, by=c("Item"))


for(id in unique(data$ID)) {
 model = (lmer(SurprisalReweighted ~ SurprisalZero + compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(ID == id, abs(compatibleSlope) < 1.5))) 
 predictability_weight = unique(((data %>% filter(ID == id))$predictability_weight))[1]
 deletion_rate = unique(((data %>% filter(ID == id))$deletion_rate))[1]
 print(paste(id, " ", summary(model)$coef[3,3], " ", deletion_rate, " ", predictability_weight))
}

#[1] "599568962   2.28385223351258   0.5   0.75"
#[1] "261611831   2.8583265121534   0.5   0"
#[1] "620654642   3.26006251282723   0.55   1"
#[1] "68680906   1.91003982838422   0.55   1"
#[1] "3535245   4.5123161510624   0.55   0.75"
#[1] "185553481   2.47574054788372   0.55   0.75"
#[1] "978406179   2.09071902101199   0.45   0"
#[1] "795928457   1.74632869742527   0.5   0.75"
#[1] "262926142   -0.344525348308368   0.45   0.5"
#[1] "69019020   2.97921990228808   0.45   0.75"
#[1] "848128467   2.9522994370413   0.45   1"
#.............



for(id in unique(data$ID)) {
 model = (lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(ID == id, abs(compatibleSlope) < 1.0))) 
 predictability_weight = unique(((data %>% filter(ID == id))$predictability_weight))[1]
 deletion_rate = unique(((data %>% filter(ID == id))$deletion_rate))[1]
 print(paste(id, " ", summary(model)$coef[2,3], " ", deletion_rate, " ", predictability_weight))
}
t.test((coef(model)$Item)$compatible.C)
t.test((zero_slopes %>% filter(abs(compatible.C) < 1.0))$compatible.C) 





# t.test((zero_slopes %>% filter(abs(compatible.C) < 1.5))$compatible.C) # already slightly > 0

for(id in unique(data$ID)) {
 model = (lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(ID == id, abs(compatibleSlope) < 1.5))) 
 predictability_weight = unique(((data %>% filter(ID == id))$predictability_weight))[1]
 deletion_rate = unique(((data %>% filter(ID == id))$deletion_rate))[1]
 print(paste(id, " ", summary(model)$coef[2,3], " ", deletion_rate, " ", predictability_weight))
}

#[1] "599568962   3.41502156729185   0.5   0.75"
#[1] "261611831   3.7205289977566   0.5   0"
#[1] "620654642   3.78739577127921   0.55   1"
#[1] "68680906   3.4683747964492   0.55   1"
#[1] "3535245   4.57283508369948   0.55   0.75"
#[1] "185553481   3.40925230680836   0.55   0.75"
#[1] "978406179   3.1302083925126   0.45   0"
#[1] "795928457   2.27522488670222   0.5   0.75"
#[1] "262926142   2.80215216826713   0.45   0.5"
#[1] "69019020   3.57669373750129   0.45   0.75"
#[1] "848128467   3.5429045338713   0.45   1"
#[1] "387749835   3.48129358876028   0.55   1"
#[1] "319356587   3.34018732357405   0.45   1"
#[1] "348524596   3.04897185643481   0.5   0"
#


t.test((zero_slopes %>% filter(compatible.C > -1.71, compatible.C < 1.87))$compatible.C)  # already slightly > 0

for(id in unique(data$ID)) {
 model = (lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(ID == id, compatibleSlope >  quantile(zero_slopes$compatible.C, 0.1)[[1]], compatibleSlope <  quantile(zero_slopes$compatible.C, 0.9)[[1]] ))) 
 predictability_weight = unique(((data %>% filter(ID == id))$predictability_weight))[1]
 deletion_rate = unique(((data %>% filter(ID == id))$deletion_rate))[1]
 print(paste(id, " ", summary(model)$coef[2,3], " ", deletion_rate, " ", predictability_weight))
}

#[1] "599568962   3.35586172854273   0.5   0.75"
#[1] "261611831   3.49529261749463   0.5   0"
#[1] "620654642   3.80068088252013   0.55   1"
#[1] "68680906   3.38376035623509   0.55   1"
#[1] "3535245   4.46711838361496   0.55   0.75"
#[1] "185553481   3.37294958527281   0.55   0.75"
#[1] "978406179   3.13519030641148   0.45   0"
#[1] "795928457   2.32331843232109   0.5   0.75"
#[1] "262926142   2.81256662349425   0.45   0.5"
#[1] "69019020   3.50228163795129   0.45   0.75"
#[1] "848128467   3.48196114820066   0.45   1"



for(id in unique(data$ID)) {
 model = (lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(ID == id, compatibleSlope > -2.2, compatibleSlope <  1.2))) 
 predictability_weight = unique(((data %>% filter(ID == id))$predictability_weight))[1]
 deletion_rate = unique(((data %>% filter(ID == id))$deletion_rate))[1]
 print(paste(id, " ", summary(model)$coef[2,3], " ", deletion_rate, " ", predictability_weight))
}

#
#
#[1] "599568962   0.677816259633849   0.5   0.75"
#[1] "261611831   1.11593683703216   0.5   0"
#[1] "620654642   1.64343754371917   0.55   1"
#[1] "68680906   0.722642729798531   0.55   1"
#[1] "3535245   1.77269661098737   0.55   0.75"
#[1] "185553481   0.921073010447262   0.55   0.75"
#[1] "978406179   0.71272355045047   0.45   0"
#[1] "795928457   0.0831893114277476   0.5   0.75"
#[1] "262926142   0.279565259901081   0.45   0.5"



model = (lmer(SurprisalReweighted ~ SurprisalZero + compatible.C + True_Minus_False.C + (1|ID) + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", deletion_rate==0.5, predictability_weight==0.0)))
#                   Estimate Std. Error t value
#(Intercept)         6.47690    0.25049  25.857
#SurprisalZero       0.42583    0.01038  41.014
#compatible.C        0.34430    0.14376   2.395
#True_Minus_False.C -0.29291    0.04267  -6.865




#                              (Intercept) SurprisalZero compatible.C True_Minus_False.C
#o_lifeguard_swimmer              7.885949     0.4258318  -2.42106069         -0.2929051
#v_psychiatrist_nurse             7.563804     0.4258318  -2.13833564         -0.2929051
#v_agent_fbi                      4.894537     0.4258318  -1.87539233         -0.2929051
#o_bureaucrat_guard               8.681734     0.4258318  -1.80389882         -0.2929051
#v_guest_cousin                   9.232113     0.4258318  -1.74232711         -0.2929051
#v_thief_detective                7.632498     0.4258318  -1.70729473         -0.2929051
#o_commander_president            9.254365     0.4258318  -1.67154452         -0.2929051
#v_manager_boss                   7.350575     0.4258318  -1.41585687         -0.2929051
#v_guest_thug                     8.260741     0.4258318  -1.40730144         -0.2929051
#v_driver_guide                   5.395718     0.4258318  -0.88509547         -0.2929051
#v_victim_swimmer                 7.210480     0.4258318  -0.81191376         -0.2929051
#o_ceo_employee                   5.088212     0.4258318  -0.62879480         -0.2929051
#v_customer_vendor                3.942342     0.4258318  -0.61683283         -0.2929051
#o_daughter_sister                7.554661     0.4258318  -0.61290560         -0.2929051
#....
#o_trickster_woman               10.992172     0.4258318   1.51715284         -0.2929051
#o_student_bully                  8.429515     0.4258318   1.51737836         -0.2929051
#v_senator_diplomat               7.969368     0.4258318   1.55327591         -0.2929051
#o_mobster_media                  6.823972     0.4258318   1.62716760         -0.2929051
#o_runner_psychiatrist            4.219955     0.4258318   1.72084452         -0.2929051
#o_criminal_officer               5.483137     0.4258318   1.76142590         -0.2929051
#v_captain_crew                   5.136621     0.4258318   1.93441487         -0.2929051
#o_trader_businessman             4.251074     0.4258318   2.12670736         -0.2929051
#v_sponsor_musician               7.085846     0.4258318   2.35495400         -0.2929051
#o_violinist_sponsors             4.261199     0.4258318   2.41054897         -0.2929051
#o_surgeon_patient                4.776588     0.4258318   2.79376405         -0.2929051



model2 = (lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1|ID) + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", deletion_rate==0.5, predictability_weight==0.0)))
#                   Estimate Std. Error t value
#(Intercept)         9.90786    0.41429  23.915
#compatible.C        0.29738    0.17382   1.711
#True_Minus_False.C -0.23427    0.04523  -5.179

library(ggrepel)
model2 = (lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", ID==167196287)))
u = coef(model2)$Item
u[order(u$compatible.C),]
u$Item = rownames(u)
plot = ggplot(u, aes(x=compatible.C)) + geom_histogram() + geom_text_repel(aes(label=Item, y=as.numeric(as.factor(Item))/5)) + theme_bw()
ggsave(plot, file="figures/analyze_L_167196287.R_slopes_hist.pdf", height=8, width=8)



# for each model ID, record the t value of the compatibility effect
for(id in unique(data$ID)) {
 model = (lmer(SurprisalReweighted ~ compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(ID == id))) 
 print(paste(id, " ", summary(model)$coef[2,3]))
}

#[1] "599568962   1.81698781031058"
#[1] "261611831   1.89260537968889"
#[1] "620654642   2.56140360164083"
#[1] "68680906   1.84793512931537"
#[1] "3535245   2.6272043692303"       ---------------------
#[1] "185553481   1.93039654881505"
#[1] "978406179   1.53456398981067"
#[1] "795928457   1.28322719538367"
#[1] "262926142   1.53471909692088"
#[1] "69019020   1.95210800028056"
#[1] "848128467   2.00834305804354"
#[1] "387749835   1.84459247909315"
#[1] "319356587   1.76337426408085"
#[1] "348524596   1.52258880441436"
#[1] "192234817   1.58290027827795"
#[1] "96253495   1.64333170042388"
#[1] "266998840   2.06483422101608"
#[1] "167196287   2.91505144572636"    --------------------------
#[1] "374361203   0.501148385133988"
#[1] "983021535   2.66896884950812" ---------------------------
#[1] "558918113   2.99936458021164" ---------------------
#[1] "198947364   1.32172033341695"
#[1] "798647547   1.86346514508168"
#[1] "781374715   2.3505506378365"


