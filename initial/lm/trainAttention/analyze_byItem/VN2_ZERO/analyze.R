
library(ggplot2)
library(tidyr)
library(dplyr)
library(lme4)

                                                                            
counts = unique(read.csv("~/scr/CODE/forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", sep="\t"))                                                                                       
counts$Ratio = log(counts$CountThat) - log(counts$CountBare)                                                                    

gulo = read.csv("/u/scr/mhahn/log-gulordava.tsv", sep="\t")


write.table(gulo %>% group_by(Noun, Region, Condition) %>% summarise(Surprisal = mean(Surprisal)), file="output/gulordava-means.tsv")

txl = read.csv("/u/scr/mhahn/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN2Stims_3_W_TXL_ZERO.py_858805767_ZeroLoss", sep="\t")
txl = merge(txl, counts, by=c("Noun"), all.x=TRUE)

gpt2m = read.csv("/sailhome/mhahn/scr/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN2Stims_3_W_GPT2M_ZERO.py_503159126_ZeroLoss", sep="\t")
gpt2m = merge(gpt2m, counts, by=c("Noun"), all.x=TRUE)

gpt2xl = read.csv("/sailhome/mhahn/scr/reinforce-logs-both-short/full-logs-tsv-perItem/char-lm-ud-stationary_12_SuperLong_WithAutoencoder_WithEx_Samples_Short_Combination_Subseq_VeryLong_WithSurp12_NormJudg_Short_Cond_Shift_NoComma_Bugfix_VN2Stims_3_W_GPT2XL_ZERO.py_931606655_ZeroLoss", sep="\t")
gpt2xl = merge(gpt2xl, counts, by=c("Noun"), all.x=TRUE)

model = lmer(SurprisalReweighted ~ RC + compatible + (1|Noun) + (1+compatible|Item), data=gulo %>% filter(!grepl("NoSC", Condition)) %>% mutate(RC= grepl("RC", Condition), compatible = grepl("_co", Condition)))
write.table(coef((model))$Item, file="output/compatibility-slots-gulordava.tsv")


slopes = coef((model))$Item
slopes$Item = rownames(slopes)
slopes$Gulordava = slopes$compatibleTRUE

slopes_ = slopes

model = lmer(SurprisalReweighted ~ Ratio + RC + compatible + (1|Noun) + (1+compatible|Item), data=txl %>% filter(!grepl("NoSC", Condition)) %>% mutate(RC= grepl("RC", Condition), compatible = grepl("_co", Condition)))
write.table(coef((model))$Item, file="output/compatibility-slots-txl.tsv")
#               Estimate Std. Error  t value
#(Intercept)     7.80578    0.49539   15.757
#Ratio           0.05942    0.04078    1.457
#RCTRUE         -1.82380    0.01757 -103.807
#compatibleTRUE  0.83921    0.22947    3.657

slopes = coef((model))$Item
slopes$Item = rownames(slopes)
slopes$TXL = slopes$compatibleTRUE

slopes_ = merge(slopes, slopes_, by=c("Item"))

model = lmer(SurprisalReweighted ~ Ratio + RC + compatible + (1|Noun) + (1+compatible|Item), data=gpt2m %>% filter(!grepl("NoSC", Condition)) %>% mutate(RC= grepl("RC", Condition), compatible = grepl("_co", Condition)))
write.table(coef((model))$Item, file="output/compatibility-slots-gpt2m.tsv")
#               Estimate Std. Error t value
#(Intercept)     6.54328    0.45363  14.424
#Ratio          -0.01343    0.03460  -0.388
#RCTRUE         -0.02291    0.01303  -1.759
#compatibleTRUE -0.19143    0.17341  -1.104


slopes = coef((model))$Item
slopes$Item = rownames(slopes)
slopes$GPT2M = slopes$compatibleTRUE

slopes_ = merge(slopes, slopes_, by=c("Item"))


model = lmer(SurprisalReweighted ~ Ratio + RC + compatible + (1|Noun) + (1+compatible|Item), data=gpt2xl %>% filter(!grepl("NoSC", Condition)) %>% mutate(RC= grepl("RC", Condition), compatible = grepl("_co", Condition)))
write.table(coef((model))$Item, file="output/compatibility-slots-gpt2xl.tsv")
#               Estimate Std. Error t value
#(Intercept)     6.87915    0.47151  14.589
#Ratio          -0.02671    0.03868  -0.691
#RCTRUE         -0.22460    0.01409 -15.939
#compatibleTRUE -0.09884    0.17767  -0.556




slopes = coef((model))$Item
slopes$Item = rownames(slopes)
slopes$GPT2XL = slopes$compatibleTRUE

slopes_ = merge(slopes, slopes_, by=c("Item"))





