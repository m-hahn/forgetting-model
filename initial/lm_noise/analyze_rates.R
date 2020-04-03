data = read.csv("perWordRates.tsv", sep="\t")


library(tidyr)
library(dplyr)
library(ggplot2)






data_ = data %>% filter(filenum == 216454325, position<20)

ggplot(data_, aes(x=position, y=score, group=counter, color=counter)) + geom_line() + facet_wrap(~word)


ggplot(data_, aes(x=position, y=score, group=word, color=word)) + geom_line() + facet_wrap(~counter)

#ggplot(data_, aes(x=position, y=score, group=word, color=word)) + geom_smooth() + facet_wrap(~counter)

data_ = data %>% filter(!(version %in% c("10_c_SuperLong_Entrop.py", "10_c_SuperLong_EntropDecay.py")))


nouns = c("fact ", "information ", "report ", "belief ", "finding ", "prediction ")
prepositions = c("by ", "about ", "of ")
data_ = data_ %>% mutate(wordc = as.character(word)) %>% mutate(pos = ifelse(wordc %in% nouns, "NOUN", ifelse(wordc %in% prepositions, "PREP", wordc)))


ggplot(data_ %>% filter(rate==-1.0, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~performance+version+filenum+learning_rate+entropy_weight)

ggplot(data_ %>% filter(rate==-1.0, counter==40, entropy_weight==0.0, memRate > 0.48, memRate < 0.52) %>% group_by(memRate, position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~(-memRate)+performance+version+filenum+learning_rate)

# long & 0.7
ggplot(data_ %>% filter(rate==-1.0, counter==200, entropy_weight==0.0, memRate > 0.28, memRate < 0.32) %>% group_by(memRate, position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~(-memRate)+performance+version+filenum+learning_rate)

ggplot(data_ %>% filter(rate==-1.0, counter==40, entropy_weight==0.0, memRate > 0.28, memRate < 0.32) %>% group_by(memRate, position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~(-memRate)+performance+version+filenum+learning_rate)

ggplot(data_ %>% filter(rate==-1.0, counter==40, entropy_weight==0.0, memRate > 0.28, memRate < 0.32) %>% group_by(memRate, position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~(-memRate)+performance+version+filenum+learning_rate)

ggplot(data_ %>% filter(rate==-1.0, counter==40, entropy_weight==0.0, memRate > 0.28, memRate < 0.32) %>% group_by(position, pos) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line()


ggplot(data_ %>% filter(rate==-1.0, counter==40, entropy_weight==0.0, memRate > 0.38, memRate < 0.42) %>% group_by(memRate, position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~(-memRate)+performance+version+filenum+learning_rate)


ggplot(data_ %>% filter(rate==-1.0, counter==40, entropy_weight==0.0, memRate > 0.18, memRate < 0.22) %>% group_by(memRate, position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~(-memRate)+performance+version+filenum+learning_rate)

ggplot(data_ %>% filter(rate==-1.0, counter==40, entropy_weight==0.0, memRate > 0.18, memRate < 0.22) %>% group_by(memRate, position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~version+filenum)

ggplot(data_ %>% filter(rate==-1.0, counter==200, entropy_weight==0.0, memRate > 0.18, memRate < 0.22) %>% group_by(memRate, position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~(-memRate)+performance+version+filenum+learning_rate)



ggplot(data_ %>% filter(grepl("12_d", version), rate==-1.0, counter==40, entropy_weight==0.0, memRate > 0.18, memRate < 0.22) %>% group_by(predictionLoss, memRate, position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~predictionLoss)

ggplot(data_ %>% filter(grepl("12_e", version), rate==-1.0, counter==40, entropy_weight==0.0, memRate > 0.18, memRate < 0.22) %>% group_by(predictionLoss, memRate, position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~predictionLoss)




ggplot(data_ %>% filter(rate==-1.0, counter==40, entropy_weight==0.0, performance>4.1) %>% group_by(memRate, position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~(-memRate)+performance+version+filenum+learning_rate)




ggplot(data_ %>% filter(rate==0.2, counter==40, entropy_weight>0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~performance+version+filenum+learning_rate+entropy_weight)


ggplot(data_ %>% filter(rate==0.2, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~performance+version+filenum+learning_rate+entropy_weight)

ggplot(data_ %>% filter(rate==0.2, counter==40, entropy_weight==0.0), aes(x=position, y=score, group=word, color=pos)) + geom_line() + facet_wrap(~performance+version+filenum+learning_rate+entropy_weight)


ggplot(data_ %>% filter(rate==0.4, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~performance+version+filenum+learning_rate+entropy_weight)

ggplot(data_ %>% filter(rate==0.5, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~performance+version+filenum+learning_rate+entropy_weight)

ggplot(data_ %>% filter(rate==0.6, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~performance+version+filenum+learning_rate+entropy_weight)

ggplot(data_ %>% filter(rate==0.6, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)) %>% group_by(position, pos) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line()


ggplot(data_ %>% filter(rate==0.7, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~performance+version+filenum+learning_rate+entropy_weight)

ggplot(data_ %>% filter(rate==0.7, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)) %>% group_by(position, pos) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line()


ggplot(data_ %>% filter(rate==0.9, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~performance+version+filenum+learning_rate+entropy_weight)

ggplot(data_ %>% filter(rate==0.9, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)) %>% group_by(position, pos) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line()


ggplot(data_ %>% filter(rate==1.0, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~performance+version+filenum+learning_rate+entropy_weight)

ggplot(data_ %>% filter(rate==1.0, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)) %>% group_by(position, pos) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line()


ggplot(data_ %>% filter(rate==1.1, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~performance+version+filenum+learning_rate+entropy_weight)

ggplot(data_ %>% filter(rate==1.1, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)) %>% group_by(position, pos) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line()


ggplot(data_ %>% filter(rate==1.2, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line() + facet_wrap(~performance+version+filenum+learning_rate+entropy_weight)

ggplot(data_ %>% filter(rate==1.2, counter==40, entropy_weight==0.0) %>% group_by(position, pos, performance, version, filenum, learning_rate, entropy_weight) %>% summarise(score=mean(score)) %>% group_by(position, pos) %>% summarise(score=mean(score)), aes(x=position, y=score, group=pos, color=pos)) + geom_line()



ggplot(data_ %>% filter(rate==0.2, counter==40), aes(x=position, y=score, group=word, color=word)) + geom_line() + facet_wrap(~performance+version+filenum+learning_rate+entropy_weight)


ggplot(data_ %>% filter(counter==40), aes(x=position, y=score, group=word, color=word)) + geom_line() + facet_wrap(~rate+performance+version+filenum)


ggplot(data_ %>% filter(counter==40), aes(x=position, y=score, group=word, color=word)) + geom_line() + facet_wrap(~rate+performance+learning_rate+entropy_weight)


ggplot(data_ %>% filter(counter==200), aes(x=position, y=score, group=word, color=word)) + geom_line() + facet_wrap(~rate+performance+version+filenum)

