data = read.csv("/u/scr/mhahn/log-gulordava.tsv", sep="\t")
library(tidyr)
library(dplyr)

write.table(data %>% group_by(Noun, Region, Condition) %>% summarise(Surprisal=mean(Surprisal)), file="output/processGurlordava_VN.R.tsv", sep="\t")



counts = unique(read.csv("~/scr/CODE/forgetting/corpus_counts/wikipedia/results/counts4NEW_Processed.tsv", sep="\t"))
counts$Ratio = log(counts$CountThat) - log(counts$CountBare)


data = merge(data, counts, by=c("Noun"))

data$SC = !grepl("NoSC", data$Condition)
data$RC = grepl("RC", data$Condition)
data$comp = grepl("_co", data$Condition)
data$SC = data$SC - mean(data$SC)
data$RC = data$RC - mean(data$RC)
data$comp = data$comp - mean(data$comp)
data[!data$SC>0,]$comp = 0
data[!data$SC>0,]$RC = 0

library(lme4)
summary(lmer(Surprisal ~ RC + SC + Ratio*SC + Ratio*RC + comp*Ratio + RC*comp + (1|Item) + (1|Noun), data=data))
#            Estimate Std. Error t value
#(Intercept) 10.53661    0.77054  13.674
#RC          -0.33663    0.08712  -3.864
#SC          -0.92132    0.07683 -11.992
#Ratio        0.16483    0.07748   2.127
#comp        -0.11187    0.08801  -1.271
#SC:Ratio    -0.62094    0.02981 -20.830
#RC:Ratio    -0.09681    0.03380  -2.864
#Ratio:comp  -0.04012    0.03380  -1.187
#RC:comp      0.04031    0.07505   0.537



