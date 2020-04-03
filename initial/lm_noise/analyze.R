data = read.csv("tableSearchResults.tsv", sep="\t")


library(ggplot2)

require(ggrepel)

ggplot(data, aes(x=memRate, y=predictionLoss, group=version, color=version)) + geom_smooth(method="loess")

library(dplyr)
library(tidyr)

ggplot(data %>% filter(grepl("Long", version)), aes(x=memRate, y=predictionLoss, group=version, color=version)) + geom_smooth(method="loess")


ggplot(data %>% filter(grepl("Long", version)), aes(x=memRate, y=predictionLoss, group=version, color=version)) + geom_smooth(method="gam", se=F)

ggplot(data %>% filter(grepl("Long", version)), aes(x=memRate, y=predictionLoss, group=version, color=version)) + geom_point()

ggplot(data %>% filter(grepl("10_c_Long", version) | grepl("12", version)), aes(x=memRate, y=predictionLoss, group=version, color=version)) + geom_point()


ggplot(data %>% filter(grepl("12", version)), aes(x=memRate, y=predictionLoss, group=version, color=version)) + geom_point()



ggplot(data %>% filter(!grepl("Long", version)), aes(x=memRate, y=predictionLoss, group=version, color=version)) + geom_smooth(method="loess")


ggplot(data %>% filter(grepl("10", version)), aes(label=version, x=memRate, y=predictionLoss, group=version, color=version)) + geom_smooth(method="loess")

ggplot(data %>% filter(grepl("10", version) | grepl("7", version)), aes(x=memRate, y=predictionLoss, group=version, color=version)) + geom_smooth(method="loess")

ggplot(data %>% filter(grepl("10_c_Long", version)), aes(label=version, x=memRate, y=predictionLoss, group=version, color=version)) + geom_smooth(method="loess")


ggplot(data %>% filter(grepl("5", version)), aes(x=memRate, y=predictionLoss, group=version, color=version)) + geom_smooth(method="loess")

ggplot(data %>% filter(grepl("7", version)), aes(x=memRate, y=predictionLoss, group=version, color=version)) + geom_smooth(method="loess")


ggplot(data %>% filter(grepl("10", version)) %>% group_by(version, rate) %>% summarise(performance=mean(performance)), aes(label=version, x=rate, y=performance, group=version, color=version)) + geom_line(method="loess")

ggplot(data  %>% filter(rate >= 0.3 & rate <= 0.7)  %>% group_by(version, rate) %>% summarise(performance=mean(performance)), aes(label=version, x=rate, y=performance, group=version, color=version)) + geom_line() +
  geom_text(data = . %>% filter(rate == "0.4"), aes(label = version, colour = version, x = 0.4, y = performance), hjust = -.1) +
  scale_colour_discrete(guide = 'none')  +    
  theme(plot.margin = unit(c(1,3,1,1), "lines")) 




ggplot(data  %>% filter(rate >= 0.3 & rate <= 0.7) %>% filter(!grepl("Long", version)) %>% group_by(version, rate) %>% summarise(performance=mean(performance)), aes(label=version, x=rate, y=performance, group=version, color=version)) + geom_line() +
  geom_text(data = . %>% filter(rate == "0.4"), aes(label = version, colour = version, x = 0.4, y = performance), hjust = -.1) +
  scale_colour_discrete(guide = 'none')  +    
  theme(plot.margin = unit(c(1,3,1,1), "lines")) 


ggplot(data  %>% filter(rate >= 0.3 & rate <= 0.7) %>% filter(grepl("10", version) & version!="10_k.py") %>% group_by(version, rate) %>% summarise(performance=mean(performance)), aes(label=version, x=rate, y=performance, group=version, color=version)) + geom_line() +
  geom_text(data = . %>% filter(rate == "0.4"), aes(label = version, colour = version, x = 0.4, y = performance), hjust = -.1) +
  scale_colour_discrete(guide = 'none')  +    
  theme(plot.margin = unit(c(1,3,1,1), "lines")) 


data_ = data %>% filter(grepl("10", version), rate==0.4, version != "10_k.py")

ggplot(data_ %>% group_by(version, learning_rate) %>% summarise(performance = mean(performance)), aes(x=log(learning_rate), y=performance, color=version)) + geom_line()

ggplot(data_ %>% filter(grepl("10_i", version)) %>% group_by(learning_rate) %>% summarise(performance = mean(performance)), aes(x=log(learning_rate), y=performance)) + geom_line()


ggplot(data_ %>% group_by(learning_rate) %>% summarise(performance = mean(performance)), aes(x=log(learning_rate), y=performance)) + geom_line()

ggplot(data %>% group_by(lr_decay) %>% summarise(performance = mean(performance)), aes(x=log(lr_decay), y=performance)) + geom_line()


ggplot(data %>% group_by(learning_rate) %>% summarise(performance = mean(performance)), aes(x=log(learning_rate), y=performance)) + geom_line()

ggplot(data %>% filter(version == "10_i.py") %>% group_by(learning_rate) %>% summarise(performance = mean(performance)), aes(x=log(learning_rate), y=performance)) + geom_line()


ggplot(data_ %>% group_by(momentum) %>% summarise(performance = mean(performance)), aes(x=momentum, y=performance)) + geom_line()


ggplot(data %>% group_by(entropy_weight) %>% summarise(performance = mean(performance)), aes(x=log(entropy_weight+1), y=performance)) + geom_line()

plot(data_$learning_rate, data_$performance)

summary(lm(performance ~ momentum + lr_decay + NUMBER_OF_REPLICATES + log(entropy_weight+1) + log(learning_rate) + version, data=data_ %>% filter(log(learning_rate) > -12)))

data__ = data %>% filter(grepl("12_Long", version), memRate > 0.15, memRate < 0.25)

summary(lm(performance ~ momentum + lr_decay + NUMBER_OF_REPLICATES + log(learning_rate), data=data__))


data__ = data %>% filter(grepl("10_c_Long", version), entropy_weight == 0)

summary(lm(performance ~ rate + momentum + lr_decay + NUMBER_OF_REPLICATES + log(learning_rate) + version, data=data__))

byPerformance = data %>% filter(rate==0.2) %>% group_by(version) %>% summarise(sd=sd(performance)/sqrt(NROW(performance)), performance=mean(performance), pessimistic=performance+2*sd)
print(byPerformance[order(byPerformance$performance),], n=50)
print(byPerformance[order(byPerformance$pessimistic),], n=50)



byPerformance = data %>% filter(rate==0.4) %>% group_by(version) %>% summarise(sd=sd(performance)/sqrt(NROW(performance)), performance=mean(performance), pessimistic=performance+2*sd)
print(byPerformance[order(byPerformance$performance),], n=50)
print(byPerformance[order(byPerformance$pessimistic),], n=50)

byPerformance = data %>% filter(rate==0.5) %>% group_by(version) %>% summarise(sd=sd(performance)/sqrt(NROW(performance)), performance=mean(performance), pessimistic=performance+2*sd)
print(byPerformance[order(byPerformance$performance),], n=50)
print(byPerformance[order(byPerformance$pessimistic),], n=50)


byPerformance = data %>% filter(rate==0.6) %>% group_by(version) %>% summarise(sd=sd(performance)/sqrt(NROW(performance)), performance=mean(performance), pessimistic=performance+2*sd)
print(byPerformance[order(byPerformance$performance),], n=50)
print(byPerformance[order(byPerformance$pessimistic),], n=50)



byPerformance = data %>% filter(rate==0.7) %>% group_by(version) %>% summarise(sd=sd(performance)/sqrt(NROW(performance)), performance=mean(performance), pessimistic=performance+2*sd)
print(byPerformance[order(byPerformance$performance),], n=50)
print(byPerformance[order(byPerformance$pessimistic),], n=50)





