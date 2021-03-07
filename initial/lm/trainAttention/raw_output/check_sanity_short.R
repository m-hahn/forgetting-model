library(tidyr)
library(dplyr)
# The LSTM by itself doesn't show much evidence of a lexical effect. Interesting contrast with the human data!
#data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Stims.py_198617139_ZeroLoss", sep=" ")
# But the Sanity version of the model clearly does
data = read.csv("/juice/scr/mhahn/reinforce-logs-both-short/full-logs-tsv/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Stims.py_2818680_Sanity", sep=" ")
counts = unique(read.csv("~/scr/CODE/forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", sep="\t"))
counts$Ratio = log(counts$CountThat) - log(counts$CountBare)


scrc = read.csv("~/scr/CODE/forgetting/corpus_counts/wikipedia/RC_annotate/results/collectResults.py.tsv", sep="\t")
scrc = scrc %>% mutate(SC_Bias = (SC+1e-10)/(SC+RC+Other+3e-10))
scrc = scrc %>% mutate(Log_SC_Bias = log(SC_Bias))

counts = merge(counts, scrc, by=c("Noun")) %>% mutate(RatioSC = Ratio + Log_SC_Bias)


data = merge(data, counts, by=c("Noun"), all.x=TRUE)

summary(lm(Surprisal ~ Ratio * Condition, data=data))
summary(lm(ThatFraction ~ Ratio, data=data %>% filter(Condition == "SC")))
u = data  %>% filter(Condition == "SC")
cor.test(u$Surprisal, u$Ratio)
cor.test(u$ThatFraction, u$Ratio)
# But really weird: u$ThatFraction just takes two values: 0.00000000 and 0.09090909. That can't be right.


