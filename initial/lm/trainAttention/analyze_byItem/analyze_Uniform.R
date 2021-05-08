data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_QC_3_W_GPT2M_UNIFORM.py_772408561_Uniform", sep="\t")
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

data[data$HasSC.C < 0,]$compatible.C = 0
data[data$HasSC.C < 0,]$HasRC.C = 0


model = lmer(SurprisalReweighted ~ HasRC.C + HasSC.C * True_Minus_False.C + compatible.C + (1+compatible.C|Item) + (1|Noun), data=data)
model2 = lmer(SurprisalReweighted ~ HasRC.C * compatible.C + HasRC.C * True_Minus_False.C + compatible.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(HasSC.C>0))


crash()

for(pred in unique(data$predictability_weight)) {
  for(del in unique(data$deletion_rate)) {
    data2 = data %>% filter(predictability_weight == pred, deletion_rate == del)
    if(nrow(data2) > 0) {
       if(length(unique(data2$ID)) == 1) {
          model = lmer(SurprisalReweighted ~ HasRC.C + HasSC.C * True_Minus_False.C + compatible.C + (1+compatible.C|Item) + (1|Noun), data=data2)
          model2 = lmer(SurprisalReweighted ~ HasRC.C * compatible.C + HasRC.C * True_Minus_False.C + compatible.C + (1+compatible.C|Item) + (1|Noun), data=data2 %>% filter(HasSC.C>0))
       } else {
          model = lmer(SurprisalReweighted ~ HasRC.C + HasSC.C * True_Minus_False.C + compatible.C + (1+compatible.C|Item) + (1|Noun) + (1+compatible.C|ID), data=data2)
          model2 = lmer(SurprisalReweighted ~ HasRC.C * compatible.C + HasRC.C * True_Minus_False.C + compatible.C + (1+compatible.C|Item) + (1|Noun) + (1+compatible.C|ID), data=data2 %>% filter(HasSC.C>0))
       }
      sink("analyze_M_QC.R.txt", append=TRUE)
       print("----")
       print(paste(pred, "  ", del))
       print(summary(model)$coef)
       print(summary(model2)$coef)
       sink()
    }
  }
}

crash()


for(id in unique(data$ID)) {
 model = (lmer(SurprisalReweighted ~ HasRC.C + compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(ID == id, HasSC.C > 0))) 
 print(paste(id, " t=", summary(model)$coef[3,3], "   ", "THAT", mean((data %>% filter(ID == id, HasSC.C>0))$ThatFractionReweighted), " ", mean((data %>% filter(ID == id))$deletion_rate), " ", mean((data %>% filter(ID == id))$predictability_weight)))
}

#[1] "74858729  t= -0.0965847666522677     THAT 49.8760619690155   0.55   0" little differentiation between the conditions
#[1] "839962200  t= 4.05629297495252     THAT 47.3447026486757   0.45   0.25"
#[1] "164077158  t= 3.04161192392009     THAT 32.0494127936032   0.55   0.5"
#[1] "900360318  t= -0.782070156063247     THAT 89.6503623188406   0.45   1"
#[1] "571568854  t= 2.64887680623599     THAT 30.3935532233883   0.55   0.25"
#[1] "575057157  t= -0.767243139858213     THAT 91.6619190404798   0.5   1" numerically looks similar for fact-like nouns, but mushed together for report-like nouns
#[1] "340958261  t= 3.88139019461207     THAT 37.9551474262869   0.5   0.25"
#[1] "856034402  t= 0.790047509768374     THAT 62.8295852073963   0.45   1" SC perfect, SCRC very bad
#[1] "786202606  t= 1.48235102180331     THAT 55.1375562218891   0.45   0" little differentiation between the conditions
#[1] "406705276  t= 2.85315520526899     THAT 58.6818465767116   0.5   0.75"
#[1] "334841046  t= 0.664298329216967     THAT 66.2674912543728   0.55   1"
#[1] "269367432  t= 1.01506949603331     THAT 71.5763993003498   0.45   1" numerically, effect only for SCRC
#[1] "118571863  t= -1.92289243463435     THAT 99.4614567716142   0.5   0.5" little differentiation
#[1] "27877760  t= 1.744134266077     THAT 69.3245252373813   0.5   1"
#[1] "54220181  t= 2.75766801447358     THAT 36.2465017491254   0.5   0.5"
#[1] "726444270  t= 0.00324040693664153     THAT 63.6557971014493   0.45   1" SC perfect, SCRC very bad
#[1] "883180537  t= 3.85469479381111     THAT 37.1547351324338   0.45   0.25"



for(id in unique(data$ID)) {
 model = (lmer(SurprisalReweighted ~  compatible.C + True_Minus_False.C + (1+compatible.C|Item) + (1|Noun), data=data %>% filter(ID == id, HasRC.C > 0))) 
 print(paste(id, " t=", summary(model)$coef[2,3], "   ", "THAT", mean((data %>% filter(ID == id, HasRC.C>0))$ThatFractionReweighted), " ", mean((data %>% filter(ID == id))$deletion_rate), " ", mean((data %>% filter(ID == id))$predictability_weight)))
}


#[1] "74858729  t= -0.40706854869121     THAT 53.8424537731134   0.55   0"
#[1] "839962200  t= 4.01174591924989     THAT 40.5156171914043   0.45   0.25"
#[1] "164077158  t= 0.254713443951974     THAT 14.5835832083958   0.55   0.5"
#[1] "900360318  t= 0.179294300258421     THAT 80.3234632683658   0.45   1"
#[1] "571568854  t= 2.73984489202737     THAT 22.5548475762119   0.55   0.25"
#[1] "575057157  t= 0.694706789869906     THAT 83.7749875062469   0.5   1"
#[1] "340958261  t= 3.9556906304858     THAT 30.1875312343828   0.5   0.25"
#[1] "856034402  t= 2.91859835068961     THAT 27.8004747626187   0.45   1"
#[1] "786202606  t= 1.97378149484259     THAT 48.8924287856072   0.45   0"
#[1] "406705276  t= 3.05705101202096     THAT 35.2744877561219   0.5   0.75"
#[1] "334841046  t= 1.08600625527248     THAT 33.8265867066467   0.55   1"
#[1] "269367432  t= 3.57655779400584     THAT 44.1965267366317   0.45   1"
#[1] "118571863  t= -1.68455490001768     THAT 99.4445277361319   0.5   0.5"
#[1] "27877760  t= 4.92147007941425     THAT 39.404047976012   0.5   1"
#[1] "54220181  t= 1.93542023707146     THAT 25.1707896051974   0.5   0.5"
#[1] "726444270  t= 1.78209531967077     THAT 28.4833833083458   0.45   1"
#[1] "883180537  t= 4.125532594836     THAT 23.0530984507746   0.45   0.25"


