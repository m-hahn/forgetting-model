data = read.csv("~/scr/reinforce-logs-both-coh/forOfPerNoun_char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Coher.py_913590423", sep="\t", header=F, quote=NULL)
names(data) <- c("Noun", "Rec", "Freq")

model = read.csv("output/analyze_Coher.R")
data = merge(data, model, by=c("Noun"))

library(tidyr)
library(dplyr)

data = data %>% group_by(Noun) %>% mutate(Freq = Freq/sum(Freq))

uthat = data %>% filter(Rec == "that")
print(cor.test(uthat$Freq, uthat$Surprisal))

