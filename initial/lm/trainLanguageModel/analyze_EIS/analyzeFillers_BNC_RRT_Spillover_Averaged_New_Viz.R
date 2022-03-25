
library(ggplot2)
library(dplyr)
library(tidyr)


Datapoints = 7271
data = read.csv("output/analyzeFillers_freq_BNC_RRT_Spillover_Averaged_New_R.tsv", sep="\t")

# This concerns nine settings with very high or very low delta, and lambda <1 (e.g., delta=0.95+lambda=0.5)
data = data %>% filter(NData == Datapoints)
doneConfigurations = paste(data$deletion_rate, data$predictability_weight)
deltas = c()
lambdas = c()
for(delta in unique(data$deletion_rate)) {
	for(lambda in unique(data$predictability_weight)) {
		if(!(paste(delta, lambda) %in% doneConfigurations)) {
			deltas = c(deltas, delta)
			lambdas = c(lambdas, lambda)
		}
	}
}
missingConfigurations = data.frame(deletion_rate = deltas, predictability_weight = lambdas)

imputed = data.frame()
for(i in (1:nrow(missingConfigurations))) {
	delta = missingConfigurations$deletion_rate[[i]]
	lambda = missingConfigurations$predictability_weight[[i]]
	to_impute = data %>% filter(abs(deletion_rate-delta) <= 0.05, abs(predictability_weight-lambda)<=0.25) %>% summarise(AIC=mean(AIC), NData=mean(NData), EIS1=mean(EIS1), EIS2=mean(EIS2), Coefficient=mean(Coefficient))
	to_impute$deletion_rate=delta
	to_impute$predictability_weight=lambda
	to_impute$X=NA
	imputed = rbind(imputed, to_impute)
}

data = rbind(data, imputed)



# Viz for all parameter configurations
plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(EIS1=mean(EIS1)), aes(x=deletion_rate, y=predictability_weight, fill=EIS1)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("") + guides(fill=guide_legend(title="EIS1")) + scale_y_reverse ()
ggsave(plot, file="figures/analyzeFillers_freq_BNC_RRT_Spillover_Averaged_New_EIS1_R.pdf", height=2, width=5)

plot = ggplot(data %>% group_by(deletion_rate, predictability_weight) %>% summarise(EIS2=mean(EIS2)), aes(x=deletion_rate, y=predictability_weight, fill=EIS2)) + geom_tile() + theme_bw() + scale_fill_gradient2() + xlab("Deletion Rate") + ylab("") + guides(fill=guide_legend(title="EIS2")) + scale_y_reverse ()
ggsave(plot, file="figures/analyzeFillers_freq_BNC_RRT_Spillover_Averaged_New_EIS2_R.pdf", height=2, width=5)


