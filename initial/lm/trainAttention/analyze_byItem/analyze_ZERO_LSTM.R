data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_J_3_W_ZERO.py_453763700_ZeroLoss", sep="\t")
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
ggsave(plot, file="figures/analyze_ZERO_LSTM.R.pdf", height=8, width=8)


model = (lmer(SurprisalReweighted ~ HasRC.C * compatible.C + HasRC.C* True_Minus_False.C +  (1+compatible.C|Item) + (1|Noun), data=data %>% filter(Region == "V1_0", HasSC.C > 0)))


library(ggrepel)
u = coef(model)$Item
u$Item = rownames(u)
plot = ggplot(u, aes(x=compatible.C)) + geom_histogram() + geom_text_repel(aes(label=Item, y=as.numeric(as.factor(Item))/5)) + theme_bw()
ggsave(plot, file="figures/analyze_ZERO_LSTM.R_slopes_hist.pdf", height=8, width=8)
write.table(u, file="analyze_ZERO_LSTM.R.tsv", sep="\t")
crash()


u2 = read.csv("analyze_ZERO_L.R.tsv", sep="\t")
u = merge(u, u2, by=c("Item"), all=TRUE)
cor.test(u$compatible.C.x, u$compatible.C.y)
#
#> cor.test(u$compatible.C.x, u$compatible.C.y)        
#
#        Pearson's product-moment correlation
#
#data:  u$compatible.C.x and u$compatible.C.y
#t = 1.88, df = 65, p-value = 0.06459
#alternative hypothesis: true correlation is not equal to 0
#95 percent confidence interval:
# -0.01387194  0.44312940
#sample estimates:
#      cor 
#0.2270934 




